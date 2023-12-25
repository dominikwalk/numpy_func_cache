import hashlib
import os
import threading
from functools import partial
from typing import Any, Callable

import numpy as np


class NumpyFuncCache:
    """
    This class provides simple (file-based) caching functionality for
    functions returning NumPy arrays.

    Author: Dominik Walk
    Created on: December 17, 2023
    """

    def __init__(self, cache_path: str) -> None:
        """
        Initializes the NumpyFuncCache class.

        Parameters:
            cache_path (str): The path where caching files should be stored.
        """
        self.cache_path = cache_path
        self.lock = threading.Lock()  # Lock for thread safety

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
        func_name = func.__name__
        args_str = ",".join(map(repr, args))
        kwargs_str = ",".join(f"{key}={value!r}" for key, value in kwargs.items())

        # Generate a unique hash for the function call to use as a filename
        input_str = f"{func_name}({args_str},{kwargs_str})"
        unique_file_name_hash = hashlib.md5(input_str.encode()).hexdigest()

        # Construct the full path to the cache file
        file_cache_path = os.path.join(self.cache_path, f"{unique_file_name_hash}.npy")

        try:
            with self.lock:  # Acquire the lock before cache access
                # Check if the cached result already exists
                if os.path.exists(file_cache_path):
                    # Load the result from the cache file
                    result = np.load(file_cache_path)
                else:
                    # Compute the result using the original function
                    result = func(*args, **kwargs)
                    # Save the result to the cache file
                    np.save(file_cache_path, result)
        except Exception as e:
            # Handle errors during the cache operation
            raise RuntimeError(f"Error during cache operation: {e}")

        return result

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
