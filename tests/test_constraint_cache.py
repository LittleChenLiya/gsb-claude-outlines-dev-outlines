"""Tests for the generation-aware constraint compilation cache.

These tests use a local, fully controlled fake compiler (no Docker, no
network) to exercise the normal path, configuration-generation switches,
concurrent calls with the same key, and failed compilations.
"""

import gc
import os
import sys
import tempfile
import threading

import pytest


@pytest.fixture
def fresh_constraint_cache():
    """Provide a fresh, isolated constraint cache for a single test."""
    for key in list(sys.modules.keys()):
        if "outlines" in key:
            del sys.modules[key]

    previous_cache_dir = os.environ.pop("OUTLINES_CACHE_DIR", None)
    with tempfile.TemporaryDirectory() as tempdir:
        os.environ["OUTLINES_CACHE_DIR"] = tempdir

        import outlines.caching

        outlines.caching.get_cache.cache_clear()
        outlines.caching.reset_constraint_cache_generation()
        outlines.caching._caching_enabled = True
        # Start with a deterministic empty cache and no lingering in-flight
        # compilations from other tests.
        outlines.caching.get_cache().clear()
        outlines.caching._inflight_compilations.clear()

        try:
            yield outlines.caching
        finally:
            # Close diskcache (releasing cache.db handles) and drop the cached
            # instance before the temporary directory is removed, otherwise
            # Windows cannot delete the open database file. gc.collect() makes
            # sure the underlying sqlite connection objects are finalized.
            outlines.caching.get_cache().close()
            outlines.caching.get_cache.cache_clear()
            outlines.caching._inflight_compilations.clear()
            gc.collect()
            del os.environ["OUTLINES_CACHE_DIR"]
            if previous_cache_dir is not None:
                os.environ["OUTLINES_CACHE_DIR"] = previous_cache_dir


class FakeConstraintCompiler:
    """Stand-in for a real (slow, external) constraint compiler."""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()

    def compile(self, constraint):
        self.calls += 1
        self.started.set()
        # Block until the test releases the compiler, which lets a second
        # thread reach the "in-flight" state deterministically.
        self.release.wait(timeout=10)
        if self.fail:
            raise RuntimeError(f"boom for {constraint}")
        return f"compiled({constraint})"

    def reset(self) -> None:
        """Reset the call counter and blocking gates for a new scenario."""
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()


def test_hit_second_call(fresh_constraint_cache):
    """The same constraint in one generation is compiled only once."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()
    fake.release.set()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 1


def test_generation_bump_misses_old_entries(fresh_constraint_cache):
    """After the configuration changes, old entries must not hit."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()
    fake.release.set()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 1

    gen_0_key = compile_constraint.__cache_key__("regex-a")
    assert caching.get_constraint_cache_generation() == 0

    # "Change the configuration": bump the generation.
    new_generation = caching.bump_constraint_cache_generation()
    assert new_generation == 1

    gen_1_key = compile_constraint.__cache_key__("regex-a")
    # The generation is part of the key, so keys differ across generations.
    assert gen_0_key != gen_1_key

    # Old entry is not matched: the constraint is recompiled.
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 2

    # The new entry itself is cached within the new generation.
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 2


def test_distinct_keys_compile_independently(fresh_constraint_cache):
    """Distinct constraints never share a cache entry."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()
    fake.release.set()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert compile_constraint("regex-b") == "compiled(regex-b)"
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 2


def test_concurrent_same_key_compiles_once(fresh_constraint_cache):
    """Two threads requesting the same key trigger a single compilation."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    results = []
    errors = []

    def worker():
        try:
            results.append(compile_constraint("regex-a"))
        except BaseException as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()

    # Wait until the first thread is inside the (blocked) fake compiler,
    # proving it holds leadership for the key.
    assert fake.started.wait(timeout=5)
    t2.start()

    # Give the second thread time to reach the in-flight wait and then let
    # the single compilation finish.
    threading.Event().wait(0.1)
    fake.release.set()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive() and not t2.is_alive()
    assert errors == []
    assert results == ["compiled(regex-a)", "compiled(regex-a)"]
    assert fake.calls == 1
    # Leadership bookkeeping is cleaned up after the compilation.
    assert caching._inflight_compilations == {}

    # A later call after both threads finished still hits the cached value.
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 1


def test_failed_compilation_does_not_pollute_cache(fresh_constraint_cache):
    """A raised exception must not be cached; retrying recompiles."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler(fail=True)

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    with pytest.raises(RuntimeError, match="boom"):
        compile_constraint("regex-a")
    assert fake.calls == 1
    assert caching._inflight_compilations == {}

    # The failure must not have populated the cache: retry recompiles.
    with pytest.raises(RuntimeError, match="boom"):
        compile_constraint("regex-a")
    assert fake.calls == 2

    # Once the compiler succeeds, the value is cached normally.
    fake.fail = False
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 3


def test_concurrent_failure_propagates_to_both_threads(fresh_constraint_cache):
    """Concurrent waiters both see the leader failure, compiled once."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler(fail=True)

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    errors = []

    def worker():
        try:
            compile_constraint("regex-a")
        except RuntimeError as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    assert fake.started.wait(timeout=5)
    t2.start()
    threading.Event().wait(0.1)
    fake.release.set()

    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive() and not t2.is_alive()
    assert len(errors) == 2
    assert all("boom" in str(error) for error in errors)
    assert fake.calls == 1
    assert caching._inflight_compilations == {}

    # Failure was not cached: a subsequent serial call recompiles.
    with pytest.raises(RuntimeError, match="boom"):
        compile_constraint("regex-a")
    assert fake.calls == 2


def test_disabled_cache_bypasses_compilation_cache(fresh_constraint_cache):
    """Disabling the cache keeps the legacy bypass behavior."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()
    fake.release.set()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    with caching.cache_disabled():
        compile_constraint("regex-a")
        compile_constraint("regex-a")

    assert fake.calls == 2

    # Outside the context, caching is enabled again.
    compile_constraint("regex-a")
    compile_constraint("regex-a")
    assert fake.calls == 3


def test_generation_switch_and_concurrency_combined(fresh_constraint_cache):
    """Concurrent compilation in one gen; a switch forces a recompile."""
    caching = fresh_constraint_cache

    fake = FakeConstraintCompiler()

    @caching.constraint_cache()
    def compile_constraint(constraint):
        return fake.compile(constraint)

    # Compile under generation 0, with two concurrent threads compiling once.
    results = []

    def worker():
        results.append(compile_constraint("regex-a"))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    assert fake.started.wait(timeout=5)
    t2.start()
    threading.Event().wait(0.1)
    fake.release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results == ["compiled(regex-a)", "compiled(regex-a)"]
    assert fake.calls == 1

    # Switch configuration: two threads in generation 1 compile once more,
    # because the generation-0 entry must not be matched.
    caching.bump_constraint_cache_generation()
    fake.reset()

    results = []

    def worker2():
        results.append(compile_constraint("regex-a"))

    t1 = threading.Thread(target=worker2)
    t2 = threading.Thread(target=worker2)
    t1.start()
    assert fake.started.wait(timeout=5)
    t2.start()
    threading.Event().wait(0.1)
    fake.release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results == ["compiled(regex-a)", "compiled(regex-a)"]
    assert fake.calls == 1

    # A follow-up serial call hits the generation-1 entry (no recompile).
    assert compile_constraint("regex-a") == "compiled(regex-a)"
    assert fake.calls == 1
