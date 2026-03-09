import hashlib
import inspect
import os
import tempfile
import threading
import multiprocessing
from functools import partial
from typing import Any, Callable, Dict, Tuple

import numpy as np


class NumpyFuncCache:
    """
    This class provides simple (file-based) caching functionality for
    functions returning NumPy arrays.

    Author: Dominik Walk
    Created on: December 17, 2023
    """

    def __init__(self, cache_path: str, thread_safety: str = "multithreading") -> None:
        """
        Initializes the NumpyFuncCache class.

        Parameters:
            cache_path (str): The path where caching files should be stored.
            thread_safety (str): Specifies the thread safety mode. 
                                 Possible values: "multithreading" or "multiprocessing".
                                 Default is "multithreading".
        """
        self.cache_path = cache_path
        self.thread_safety = thread_safety
        if thread_safety == "multiprocessing":
            self.lock = multiprocessing.Lock()  # Lock for multiprocessing safety
        else:
            self.lock = threading.Lock()  # Lock for multithreading safety
            self._key_locks: Dict[str, threading.Lock] = {}
            self._key_locks_guard = threading.Lock()

        try:
            # Create the cache directory if it does not exist
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        except OSError as e:
            raise RuntimeError(f"Error creating cache directory: {e}")

    def _compute_func_cached(
        self, func: Callable[..., np.ndarray], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Internal method for computing or retrieving function results with caching.

        Parameters:
            func (Callable): The function whose results should be cached.
            *args (Any): Positional arguments for the function.
            **kwargs (Any): Keyword arguments for the function.

        Returns:
            np.ndarray: The result of the function, either from the cache or recomputed.
        """
        hasher = hashlib.md5()
        self._update_cache_key_hash(hasher, self._get_function_fingerprint(func))
        self._update_cache_key_hash(hasher, args)
        self._update_cache_key_hash(hasher, kwargs)
        unique_file_name_hash = hasher.hexdigest()

        # Construct the full path to the cache file
        file_cache_path = os.path.join(self.cache_path, f"{unique_file_name_hash}.npy")

        try:
            lock = self._get_cache_lock(unique_file_name_hash)
            with lock:  # Acquire the lock before cache access
                # Check if the cached result already exists
                if os.path.exists(file_cache_path):
                    # Load the result from the cache file
                    result = np.load(file_cache_path)
                else:
                    # Compute the result using the original function
                    result = func(*args, **kwargs)
                    # Save the result to the cache file
                    self._save_array_atomically(file_cache_path, result)
        except Exception as e:
            # Handle errors during the cache operation
            raise RuntimeError(f"Error during cache operation: {e}")

        return result

    def _get_cache_lock(self, cache_key: str) -> Any:
        if self.thread_safety != "multithreading":
            return self.lock

        with self._key_locks_guard:
            lock = self._key_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[cache_key] = lock
            return lock

    def _save_array_atomically(self, file_cache_path: str, result: np.ndarray) -> None:
        file_name = os.path.basename(file_cache_path)
        fd, temp_file_path = tempfile.mkstemp(
            prefix=f".{file_name}.",
            suffix=".tmp",
            dir=self.cache_path,
        )
        try:
            with os.fdopen(fd, "wb") as temp_file:
                np.save(temp_file, result)
            os.replace(temp_file_path, file_cache_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _get_function_fingerprint(
        self, func: Callable[..., np.ndarray]
    ) -> Tuple[Any, ...]:
        """
        Build a deterministic fingerprint for function identity and implementation.

        This lets cache entries change when function code changes, even if the
        function name stays the same.
        """
        module_name = getattr(func, "__module__", None)
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))

        code_obj = getattr(func, "__code__", None)
        code_fingerprint = None
        if code_obj is not None:
            code_fingerprint = (
                code_obj.co_code,
                code_obj.co_consts,
                code_obj.co_names,
                code_obj.co_varnames,
                code_obj.co_argcount,
                code_obj.co_kwonlyargcount,
            )

        source_hash = None
        try:
            source = inspect.getsource(func)
            source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
        except (OSError, TypeError):
            # Source is not always available (e.g., dynamic/built-in functions).
            source_hash = None

        return (
            "func",
            module_name,
            qualname,
            code_fingerprint,
            getattr(func, "__defaults__", None),
            getattr(func, "__kwdefaults__", None),
            source_hash,
        )

    def _update_cache_key_hash(self, hasher: Any, value: Any) -> None:
        """
        Update a hash object with a deterministic representation of `value`.

        NumPy arrays are hashed by dtype, shape and raw bytes to avoid key
        collisions caused by truncated repr output for large arrays.
        """
        if isinstance(value, (bytes, bytearray, memoryview)):
            hasher.update(b"bytes:")
            hasher.update(bytes(value))
            hasher.update(b";")
            return

        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            hasher.update(b"ndarray:")
            hasher.update(str(contiguous.dtype).encode("utf-8"))
            hasher.update(b":")
            hasher.update(str(contiguous.shape).encode("utf-8"))
            hasher.update(b":")
            hasher.update(memoryview(contiguous).cast("B"))
            hasher.update(b";")
            return

        if isinstance(value, np.generic):
            self._update_cache_key_hash(hasher, np.asarray(value))
            return

        if isinstance(value, tuple):
            hasher.update(b"tuple[")
            for item in value:
                self._update_cache_key_hash(hasher, item)
                hasher.update(b",")
            hasher.update(b"]")
            return

        if isinstance(value, list):
            hasher.update(b"list[")
            for item in value:
                self._update_cache_key_hash(hasher, item)
                hasher.update(b",")
            hasher.update(b"]")
            return

        if isinstance(value, dict):
            hasher.update(b"dict{")
            for key in sorted(value.keys(), key=repr):
                self._update_cache_key_hash(hasher, key)
                hasher.update(b":")
                self._update_cache_key_hash(hasher, value[key])
                hasher.update(b",")
            hasher.update(b"}")
            return

        hasher.update(type(value).__qualname__.encode("utf-8"))
        hasher.update(b":")
        hasher.update(repr(value).encode("utf-8"))
        hasher.update(b";")

    def create_cached_func(
        self, func: Callable[..., np.ndarray]
    ) -> Callable[..., np.ndarray]:
        """
        Creates a cached version of the given function.

        Parameters:
            func (Callable): The function whose results should be cached.

        Returns:
            Callable[..., np.ndarray]: A cached version of the function.
        """
        # Create a partial function with caching behavior
        cached_compute_func = partial(
            self._compute_func_cached,
            func,
        )

        return cached_compute_func

    def clear_cache(self, remove_dir=False):
        """
        Clears the cache by deleting all files in the cache directory.

        Parameters:
            remove_dir (bool): If True, removes the entire cache directory.
        """
        try:
            with self.lock:  # Acquire the lock before cache clearing
                # Iterate over all files in the cache directory
                for file_name in os.listdir(self.cache_path):
                    file_path = os.path.join(self.cache_path, file_name)
                    try:
                        # Delete each file in the cache directory
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        # Handle errors during file deletion
                        print(f"Error deleting file {file_path}: {e}")

                # Optionally remove the entire cache directory
                if remove_dir:
                    os.rmdir(self.cache_path)
        except Exception as e:
            # Handle errors during cache clearing
            raise RuntimeError(f"Error during cache clearing: {e}")
