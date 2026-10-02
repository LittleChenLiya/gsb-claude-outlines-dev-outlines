"""Caching and memoization of function calls."""

import asyncio
import contextlib
import functools
import os
import tempfile
import threading
from typing import Any, Callable, Optional

import cloudpickle
from diskcache import Cache, Disk
from diskcache.core import ENOVAL, UNKNOWN, args_to_key, full_name

_caching_enabled = True

# Monotonically increasing generation of the constraint compilation
# configuration. The current generation is part of every constraint cache key,
# so when the configuration changes (the generation is bumped) entries stored
# under previous generations are no longer matched.
_constraint_cache_generation = 0


def bump_constraint_cache_generation() -> int:
    """Invalidate every cached constraint compilation entry.

    Bump the configuration generation used when building constraint cache
    keys. Entries produced under previous generations are left on disk but can
    no longer be hit; the next call for a given constraint compiles it again
    under the new generation.

    Returns
    -------
    int
        The new configuration generation.

    """
    global _constraint_cache_generation
    _constraint_cache_generation += 1
    return _constraint_cache_generation


def get_constraint_cache_generation() -> int:
    """Return the current configuration generation of the constraint cache."""
    return _constraint_cache_generation


def reset_constraint_cache_generation() -> None:
    """Reset the configuration generation of the constraint cache to zero.

    This is mainly useful in tests that need a deterministic starting point.
    It does not erase previously cached entries.

    """
    global _constraint_cache_generation
    _constraint_cache_generation = 0


class _InflightCompilation:
    """A compilation that one thread is running while others wait for it."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: Optional[BaseException] = None


# Maps a full constraint cache key to the in-flight compilation that computes
# its value. Guarantees that concurrent callers with the same key trigger the
# compiler only once.
_inflight_compilations: dict[Any, _InflightCompilation] = {}
_inflight_compilations_lock = threading.Lock()


class CloudpickleDisk(Disk): # pragma: no cover
    def __init__(self, directory, compress_level=1, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        data = cloudpickle.dumps(key)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return cloudpickle.loads(data)

    def store(self, value, read, key=UNKNOWN):
        if not read:
            value = cloudpickle.dumps(value)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = cloudpickle.loads(data)
        return data


@functools.lru_cache(1)
def get_cache():
    """Get the context object that contains previously-computed return values.

    The cache is used to avoid unnecessary computations and API calls, which can
    be long and expensive for large models.

    The cache directory defaults to `HOMEDIR/.cache/outlines`, but this choice
    can be overridden by the user by setting the value of the `OUTLINES_CACHE_DIR`
    environment variable.

    """
    from outlines._version import __version__ as outlines_version  # type: ignore

    outlines_cache_dir = os.environ.get("OUTLINES_CACHE_DIR")
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    home_dir = os.path.normpath(os.path.expanduser("~"))
    if outlines_cache_dir:
        # OUTLINES_CACHE_DIR takes precedence
        cache_dir = outlines_cache_dir
    elif xdg_cache_home:  # pragma: no cover
        cache_dir = os.path.join(xdg_cache_home, "outlines")
    elif home_dir != "/": # pragma: no cover
        cache_dir = os.path.join(home_dir, ".cache", "outlines")
    else:  # pragma: no cover
        # home_dir may be / inside a docker container without existing user
        tempdir = tempfile.gettempdir()
        cache_dir = os.path.join(tempdir, ".cache", "outlines")

    memory = Cache(
        cache_dir,
        eviction_policy="none",
        cull_limit=0,
        disk=CloudpickleDisk,
    )

    # ensure if version upgrade occurs, old cache is pruned
    if outlines_version != memory.get("__version__"):
        memory.clear()
    memory["__version__"] = outlines_version

    return memory


def cache(expire: Optional[float] = None, typed=False, ignore=()):
    """Caching decorator for memoizing function calls.

    The cache key is created based on the values returned by the key_function callable
    if provided or based on the arguments of the decorated function directly otherwise

    This is based on `diskcache`'s `memoize`.

    Parameters
    ----------
    expire
        Seconds until arguments expire.
    typed
        Cache different types separately.
    ignore
        Positional or keyword arguments to ignore.

    Returns
    -------
        A decorator function that can be applied to other functions.
    """

    def decorator(cached_function: Callable):
        memory = get_cache()

        base = (full_name(cached_function),)

        if asyncio.iscoroutinefunction(cached_function):  # pragma: no cover

            async def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return await cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = await cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        else:

            def wrapper(*args, **kwargs):
                if not _caching_enabled:
                    return cached_function(*args, **kwargs)

                cache_key = wrapper.__cache_key__(*args, **kwargs)
                result = wrapper.__memory__.get(cache_key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = cached_function(*args, **kwargs)
                    wrapper.__memory__.set(cache_key, result, expire, retry=True)

                return result

        def __cache_key__(*args, **kwargs):
            """Make key for cache given function arguments."""
            return args_to_key(base, args, kwargs, typed, ignore)

        wrapper.__cache_key__ = __cache_key__  # type: ignore
        wrapper.__memory__ = memory  # type: ignore
        wrapper.__wrapped__ = cached_function  # type: ignore

        return wrapper

    return decorator


def constraint_cache(expire: Optional[float] = None, typed=False, ignore=()):
    """Caching decorator for constraint compilation functions.

    Unlike `cache`, this decorator understands *configuration generations*.
    The current generation (see `bump_constraint_cache_generation`) is included
    in every cache key, which ensures that entries compiled under an older
    configuration are never matched after the configuration changes.

    It additionally provides two failure/concurrency guarantees:

    * Concurrent callers with the same arguments in the same generation compile
      the constraint only once; waiting callers reuse the leader's result.
    * If a compilation raises, nothing is written to the cache and every
      concurrent caller receives the same exception, so failed compilations can
      never pollute the cache.

    Parameters
    ----------
    expire
        Seconds until arguments expire.
    typed
        Cache different types separately.
    ignore
        Positional or keyword arguments to ignore.

    Returns
    -------
        A decorator function that can be applied to compilation functions.

    """

    def decorator(cached_function: Callable):
        memory = get_cache()

        base = (full_name(cached_function),)

        def wrapper(*args, **kwargs):
            if not _caching_enabled:
                return cached_function(*args, **kwargs)

            cache_key = wrapper.__cache_key__(*args, **kwargs)

            while True:
                # Fast path: a value compiled under the current generation
                # exists.
                result = memory.get(cache_key, default=ENOVAL, retry=True)
                if result is not ENOVAL:
                    return result

                # Either become the leader for this key or join an in-flight
                # compilation started by another thread.
                with _inflight_compilations_lock:
                    inflight = _inflight_compilations.get(cache_key)
                    if inflight is None:
                        inflight = _InflightCompilation()
                        _inflight_compilations[cache_key] = inflight
                        is_leader = True
                    else:
                        is_leader = False

                if not is_leader:
                    # Wait for the leader's compilation and reuse its result.
                    inflight.event.wait()
                    result = memory.get(cache_key, default=ENOVAL, retry=True)
                    if result is not ENOVAL:
                        return result
                    if inflight.error is not None:
                        # Share the leader's failure without retrying it here;
                        # nothing was stored in the cache.
                        raise inflight.error
                    # The leader's entry disappeared (e.g. the cache was
                    # cleared); loop around and compile again.
                    continue

                try:
                    result = cached_function(*args, **kwargs)
                except BaseException as error:
                    # Do not store anything: a failed compilation must not
                    # pollute the cache. Wake up the waiters and share the
                    # failure.
                    inflight.error = error
                    with _inflight_compilations_lock:
                        _inflight_compilations.pop(cache_key, None)
                    inflight.event.set()
                    raise

                memory.set(cache_key, result, expire, retry=True)
                with _inflight_compilations_lock:
                    _inflight_compilations.pop(cache_key, None)
                inflight.event.set()

                return result

        def __cache_key__(*args, **kwargs):
            """Make a key for the cache given function arguments.

            The current configuration generation is part of the key so that
            configuration changes make all entries of previous generations
            unmatchable.
            """
            args_with_generation = (
                ("__constraint_cache_generation__", _constraint_cache_generation),
                *args,
            )
            return args_to_key(base, args_with_generation, kwargs, typed, ignore)

        wrapper.__cache_key__ = __cache_key__  # type: ignore
        wrapper.__memory__ = memory  # type: ignore
        wrapper.__wrapped__ = cached_function  # type: ignore

        return wrapper

    return decorator


def disable_cache():
    """Disable the cache for this session.

    Generative models output different results each time they are called when
    sampling. This can be a desirable property for some workflows, in which case
    one can call `outlines.call.disable` to disable the cache for the session.

    This function does not delete the cache, call `outlines.cache.clear`
    instead. It also does not overwrite the cache with the values returned
    during the session.

    Example
    -------

    `outlines.cache.disable` should be called right after importing outlines:

    >>> import outlines.caching as cache
    >>> cache.disable_cache()

    """
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    """Erase the cache completely."""
    memory = get_cache()
    memory.clear()


@contextlib.contextmanager
def cache_disabled():
    # outlines.caching._caching_enabled
    global _caching_enabled
    original_state = _caching_enabled
    _caching_enabled = False
    try:
        yield
    finally:
        _caching_enabled = original_state
