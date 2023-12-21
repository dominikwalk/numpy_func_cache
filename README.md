# NumpyFuncCache

## Overview
`NumpyFuncCache` is a Python class that provides simple file-based caching functionality for functions returning NumPy arrays. This can be particularly useful when working with computationally expensive functions that generate the same output for a given set of input parameters. Instead of recomputing the result every time, the class caches the result in a file, allowing for faster retrieval.

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

## Important Note on Cache Size
The `NumpyFuncCache` class does not currently manage or limit the size of the cache. Depending on the size of the NumPy arrays being cached, the cache directory can potentially grow very large. Users are advised to monitor the size of the cache directory and take appropriate actions, such as manually clearing the cache or implementing additional mechanisms to manage cache size.

## How It Works
The class uses a file-based approach to caching. When a function is called, the class checks if the result for the given input parameters already exists in the cache directory. If it does, the result is loaded from the file; otherwise, the function is computed, and the result is saved to a new file in the cache.

## Example
```python
# Example usage of the NumpyFuncCache class

# Initialize the cache with a specified directory
cache = NumpyFuncCache("/path/to/cache")

# Define a function to be cached
def expensive_function(x):
    # ... (some computation)
    return result_array

# Create a cached version of the function
cached_function = cache.create_cached_func(expensive_function)

# Call the cached function
result = cached_function(input_value)

# Clear the cache (remove only files)
cache.clear_cache()

# Clear the cache and remove the entire cache directory
cache.clear_cache(remove_dir=True)
```

## Author
- Dominik Walk