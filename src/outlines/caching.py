"""Caching and memoization of function calls."""

import asyncio
import contextlib
import functools
import os
import tempfile
import threading
from typing import Callable, Optional

import cloudpickle
from diskcache import Cache, Disk
from diskcache.core import ENOVAL, UNKNOWN, args_to_key, full_name

_caching_enabled = True


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


class ConstraintCache:
    """Cache compiled constraints with a configuration generation.

    A constraint compiler turns a high-level constraint (a regex, a JSON
    schema, a context-free grammar, ...) into a compiled object that is
    expensive to build. This cache memoizes those compiled objects.

    Every cache key includes a *configuration generation*. When the
    configuration changes (a new model, a different tokenizer, ...) call
    :meth:`bump_generation`; the generation is part of every key so entries
    compiled under an older generation never match again. Old entries are
    not deleted, they simply stop matching.

    Concurrency
    -----------
    Concurrent calls with the same key only compile once. The first call
    compiles while the others wait on a per-key lock and then reuse the
    compiled result.

    Failures
    --------
    A compilation that raises is never cached: the exception propagates to
    the caller and the next call retries the compilation.
    """

    _GENERATION_KEY = "__constraint_generation__"

    def __init__(self, memory: Optional[Cache] = None):
        self.memory = memory if memory is not None else get_cache()
        self._generation = self.memory.get(self._GENERATION_KEY, 0)
        self._locks: dict = {}
        self._locks_guard = threading.Lock()

    @property
    def generation(self) -> int:
        """The current configuration generation."""
        return self._generation

    def bump_generation(self) -> int:
        """Advance the configuration generation.

        Returns the new generation. Old cache entries no longer match
        because the generation is part of every cache key.
        """
        with self._locks_guard:
            self._generation += 1
            self.memory.set(self._GENERATION_KEY, self._generation)
        return self._generation

    def _make_key(self, compiler: Callable, constraint, *args, **kwargs):
        if hasattr(compiler, "__qualname__"):
            name = full_name(compiler)
        else:
            # callable instance: identify it by its class
            name = full_name(type(compiler))
        base = (name,)
        return args_to_key(base, (constraint,) + args, kwargs, False, ())

    def compile(self, compiler: Callable, constraint, *args, **kwargs):
        """Compile ``constraint`` with ``compiler``, caching the result.

        The compiler is called as ``compiler(constraint, *args, **kwargs)``.
        """
        if not _caching_enabled:
            return compiler(constraint, *args, **kwargs)

        base_key = self._make_key(compiler, constraint, *args, **kwargs)
        key = base_key + (self._generation,)

        result = self.memory.get(key, default=ENOVAL, retry=True)
        if result is not ENOVAL:
            return result

        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock

        with lock:
            # Another thread may have compiled while we were waiting.
            result = self.memory.get(key, default=ENOVAL, retry=True)
            if result is not ENOVAL:
                return result
            # A failed compilation must not pollute the cache: only set the
            # entry after the compiler returned successfully.
            result = compiler(constraint, *args, **kwargs)
            self.memory.set(key, result, retry=True)
            return result


@functools.lru_cache(1)
def get_constraint_cache() -> ConstraintCache:
    """Get the shared constraint compilation cache."""
    return ConstraintCache()
