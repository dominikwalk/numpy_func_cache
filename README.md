# NumpyFuncCache

## Overview
`NumpyFuncCache` is a Python class that provides simple file-based caching functionality for functions returning NumPy arrays. This can be particularly useful when working with computationally expensive functions that generate the same output for a given set of input parameters. Instead of recomputing the result every time, the class caches the result in a file, allowing for faster retrieval.

## Limits and Important Notes

### Function Hashing
In the interests of simplicity, the `NumpyFuncCache` class calculates the hash for caching based on the function name and input parameters, **making it vulnerable to changes to the internal structure or calculations of the function**. If there are changes to the function, **users are responsible for manually clearing the cache to ensure the accuracy of the cached results.**

### Cache Size
The class does not currently manage or limit the size of the cache. Depending on the size of cached NumPy arrays, the cache directory may grow large over time. Users are advised to monitor the cache directory size and take appropriate actions, such as clearing the cache or implementing additional mechanisms to manage cache size.

### Thread Safety
The `NumpyFuncCache` class is thread-safe, ensuring that cache operations are atomic and prevent race conditions in a multithreaded environment.

## Usage

### Initialization
```python
from NumpyFuncCache.numpy_func_cache import NumpyFuncCache

# Specify the path where caching files should be stored
cache_path = "/path/to/cache/directory"
cache = NumpyFuncCache(cache_path)
```

### Caching a Function
```python
import numpy as np

# Define a function that returns a NumPy array
def my_expensive_function(x):
    # ... (some computation)
    return result_array

# Create a cached version of the function
cached_function = cache.create_cached_func(my_expensive_function)

# Call the cached function as you would the original function
result = cached_function(input_parameter)
```

### Clearing the Cache
```python
# Clear the cache (remove only files)
cache.clear_cache()

# Clear the cache and remove the entire cache directory
cache.clear_cache(remove_dir=True)
```

## How It Works
The class uses a file-based approach to caching. When a function is called, the class checks if the result for the given input parameters already exists in the cache directory. If it does, the result is loaded from the file; otherwise, the function is computed, and the result is saved to a new file in the cache.

## Example
```python
# Example usage of the NumpyFuncCache class
import numpy as np
from NumpyFuncCache.numpy_func_cache import NumpyFuncCache

# Initialize the cache with a specified directory
cache = NumpyFuncCache(".np_func_cache")


# Define a function to be cached
def expensive_function(x, y=10):
    # ... (some computation)
    result_array = np.linspace(x, y)
    return result_array


# Create a cached version of the function
cached_expensive_function = cache.create_cached_func(expensive_function)

# Call the original function
original_result = expensive_function(7, y=42)

# Call the cached function
cached_result = cached_expensive_function(7, y=42)

# Both results should be the same NumPy array
assert np.array_equal(original_result, cached_result)

# Clear the cache (remove only files)
cache.clear_cache()

# Clear the cache and remove the entire cache directory
cache.clear_cache(remove_dir=True)
```

## Author
- Dominik Walk
