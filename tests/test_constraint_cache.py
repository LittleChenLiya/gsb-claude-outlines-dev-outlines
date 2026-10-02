"""Tests for the constraint compilation cache.

The fake compiler is a controllable fixture that tracks the number of
compilations, blocks threads on a gate, switches its configuration, and
fails on demand. This lets us deterministically exercise:

- the normal path (compile once, cache hit on the second call),
- the configuration-generation path (bump the generation, old entries miss),
- the concurrency path (two threads, only one compilation),
- the failure path (a failed compilation is not cached, retry succeeds),
- the compatibility path (``cache_disabled`` bypasses the cache).
"""

import threading

import pytest

from outlines.caching import ConstraintCache, cache_disabled


class FakeCompiler:
    """A controllable fake constraint compiler.

    Tracks the number of compilations, can block threads on a gate, switch
    its configuration, and fail on demand.
    """

    def __init__(self):
        self.calls = 0
        self.config = "v1"
        self.fail = False
        self._gate = threading.Event()
        self._entered = threading.Event()

    def __call__(self, constraint):
        return self.compile(constraint)

    def compile(self, constraint):
        self.calls += 1
        self._entered.set()
        self._gate.wait()
        if self.fail:
            raise RuntimeError(f"compilation of {constraint!r} failed")
        return f"compiled({constraint}, {self.config})"

    def release(self):
        self._gate.set()

    def wait_until_entered(self, timeout=5):
        self._entered.wait(timeout)


@pytest.fixture
def constraint_cache():
    import gc
    import tempfile

    from diskcache import Cache

    with tempfile.TemporaryDirectory() as tempdir:
        memory = Cache(tempdir, eviction_policy="none", cull_limit=0)
        try:
            yield ConstraintCache(memory)
        finally:
            memory.close()
            del memory
            gc.collect()


def test_compile_caches_result(constraint_cache):
    compiler = FakeCompiler()
    compiler.release()  # no blocking

    result1 = constraint_cache.compile(compiler, "regex")
    assert result1 == "compiled(regex, v1)"
    assert compiler.calls == 1

    # second call is a cache hit: the compiler is not called again
    result2 = constraint_cache.compile(compiler, "regex")
    assert result2 == result1
    assert compiler.calls == 1


def test_generation_bump_invalidates_old_entries(constraint_cache):
    compiler = FakeCompiler()
    compiler.release()

    result1 = constraint_cache.compile(compiler, "regex")
    assert result1 == "compiled(regex, v1)"
    assert compiler.calls == 1

    # switch configuration and bump the generation
    compiler.config = "v2"
    constraint_cache.bump_generation()

    # the old entry must not hit: recompiled under the new configuration
    result2 = constraint_cache.compile(compiler, "regex")
    assert result2 == "compiled(regex, v2)"
    assert compiler.calls == 2

    # the new entry is now cached
    result3 = constraint_cache.compile(compiler, "regex")
    assert result3 == "compiled(regex, v2)"
    assert compiler.calls == 2


def test_concurrent_compile_only_compiles_once(constraint_cache):
    compiler = FakeCompiler()
    results = []

    def worker():
        results.append(constraint_cache.compile(compiler, "regex"))

    thread_a = threading.Thread(target=worker)
    thread_b = threading.Thread(target=worker)

    # start A; it enters the compiler and blocks on the gate
    thread_a.start()
    compiler.wait_until_entered()

    # start B; it must wait on the per-key lock, not enter the compiler
    thread_b.start()
    thread_b.join(timeout=1)
    assert thread_b.is_alive()  # B is blocked on the lock
    assert compiler.calls == 1  # only A entered the compiler

    # release the gate: A finishes, caches, B gets a cache hit
    compiler.release()
    thread_a.join()
    thread_b.join()

    assert compiler.calls == 1
    assert results == ["compiled(regex, v1)", "compiled(regex, v1)"]


def test_failed_compile_is_not_cached(constraint_cache):
    compiler = FakeCompiler()
    compiler.release()
    compiler.fail = True

    # first call fails: the exception propagates, nothing is cached
    with pytest.raises(RuntimeError, match="failed"):
        constraint_cache.compile(compiler, "regex")
    assert compiler.calls == 1

    # second call retries (the failure was not cached)
    with pytest.raises(RuntimeError, match="failed"):
        constraint_cache.compile(compiler, "regex")
    assert compiler.calls == 2

    # fix the compiler: the next call succeeds and is cached
    compiler.fail = False
    result = constraint_cache.compile(compiler, "regex")
    assert result == "compiled(regex, v1)"
    assert compiler.calls == 3

    # subsequent call is a cache hit
    result2 = constraint_cache.compile(compiler, "regex")
    assert result2 == result
    assert compiler.calls == 3


def test_cache_disabled_bypasses_cache(constraint_cache):
    compiler = FakeCompiler()
    compiler.release()

    constraint_cache.compile(compiler, "regex")
    assert compiler.calls == 1

    with cache_disabled():
        result = constraint_cache.compile(compiler, "regex")
        assert result == "compiled(regex, v1)"
        assert compiler.calls == 2  # compiler called again, cache bypassed

    # the cache is re-enabled on scope exit: hit again
    constraint_cache.compile(compiler, "regex")
    assert compiler.calls == 2


def test_get_constraint_cache_returns_shared_instance():
    from outlines.caching import get_constraint_cache

    assert get_constraint_cache() is get_constraint_cache()
