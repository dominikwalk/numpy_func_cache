import os
import tempfile
import numpy as np
import pytest
from NumpyFuncCache.numpy_func_cache import NumpyFuncCache


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
