import os
import tempfile
import numpy as np
import threading
import multiprocessing
import pytest

from numpy_func_cache import NumpyFuncCache


# Fixture to create a temporary directory for testing
@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir


def test_cache_creation(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)
    assert os.path.exists(temp_cache_dir)


def test_cached_func(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)

    # Define a sample function for testing
    def sample_func(x):
        return np.array([x, x**2, x**3])

    # Create a cached version of the function
    cached_func = cache.create_cached_func(sample_func)

    # Test with different inputs
    result1_cached = cached_func(2)
    result1_not_cached = sample_func(2)
    assert np.array_equal(result1_not_cached, result1_cached)

    result2_cached = cached_func(np.array([1, 2, 3]))
    result2_not_cached = sample_func(np.array([1, 2, 3]))
    assert np.array_equal(result2_cached, result2_not_cached)

    # Ensure the cache directory contains the expected number of files
    cache_files = os.listdir(temp_cache_dir)
    assert len(cache_files) == 2  # One file for each function call


def test_large_numpy_args_do_not_collide_on_repr(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)
    call_count = {"count": 0}

    def sample_func(x):
        call_count["count"] += 1
        return np.array([int(np.sum(x))])

    cached_func = cache.create_cached_func(sample_func)

    arr1 = np.concatenate([np.arange(10), np.ones(2000, dtype=int), np.arange(10)])
    arr2 = np.concatenate([np.arange(10), np.full(2000, 2, dtype=int), np.arange(10)])

    # Demonstrates why repr-based hashing is unsafe for large arrays.
    assert repr(arr1) == repr(arr2)

    result1 = cached_func(arr1)
    result2 = cached_func(arr2)

    assert not np.array_equal(result1, result2)
    assert call_count["count"] == 2
    assert len(os.listdir(temp_cache_dir)) == 2


def test_function_implementation_change_uses_new_cache_entry(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)
    namespace = {"np": np, "__name__": "dynamic_test_module"}

    exec("def sample_func(x):\n    return np.array([x + 1])", namespace)
    cached_func_v1 = cache.create_cached_func(namespace["sample_func"])
    result_v1 = cached_func_v1(3)

    exec("def sample_func(x):\n    return np.array([x + 2])", namespace)
    cached_func_v2 = cache.create_cached_func(namespace["sample_func"])
    result_v2 = cached_func_v2(3)

    assert np.array_equal(result_v1, np.array([4]))
    assert np.array_equal(result_v2, np.array([5]))
    assert len(os.listdir(temp_cache_dir)) == 2


def test_kwargs_order_uses_same_cache_entry(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)
    call_count = {"count": 0}

    def sample_func(a, b):
        call_count["count"] += 1
        return np.array([a, b, a + b])

    cached_func = cache.create_cached_func(sample_func)

    result1 = cached_func(a=1, b=2)
    result2 = cached_func(b=2, a=1)

    assert np.array_equal(result1, result2)
    assert call_count["count"] == 1
    assert len(os.listdir(temp_cache_dir)) == 1

def test_cache_clearing(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)

    # Define a sample function for testing
    def sample_func(x):
        return np.array([x, x**2, x**3])

    # Create a cached version of the function
    cached_func = cache.create_cached_func(sample_func)

    # Test with different inputs
    result1 = cached_func(2)
    result2 = cached_func(3)

    # Clear the cache
    cache.clear_cache()

    # Ensure the cache directory is empty after clearing
    cache_files = os.listdir(temp_cache_dir)
    assert not cache_files


def test_cache_clearing_with_dir_removal(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)

    # Define a sample function for testing
    def sample_func(x):
        return np.array([x, x**2, x**3])

    # Create a cached version of the function
    cached_func = cache.create_cached_func(sample_func)

    # Test with different inputs
    result1 = cached_func(2)
    result2 = cached_func(3)

    # Clear the cache and remove the directory
    cache.clear_cache(remove_dir=True)

    # Ensure the cache directory is removed
    assert not os.path.exists(temp_cache_dir)


# Define a sample function for testing
def sample_func(x=42):
    return np.array([x, x**2, x**3])


def test_multithreading_safety(temp_cache_dir):
    cache = NumpyFuncCache(temp_cache_dir)

    # Create a cached version of the function
    cached_func = cache.create_cached_func(sample_func)

    # Start multiple threads to execute the test_caching function
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=cached_func)
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


def test_multiprocessing_safety(temp_cache_dir):
    cache = NumpyFuncCache(
        temp_cache_dir, thread_safety="multiprocessing"
    )  # Set to multiprocessing safety

    # Create a cached version of the function
    cached_func = cache.create_cached_func(sample_func)

    # Create multiple processes to execute the cached function
    processes = []
    result_queue = multiprocessing.Queue()
    for _ in range(5):
        process = multiprocessing.Process(target=cached_func)
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
