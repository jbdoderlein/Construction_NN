import c_extension
import time
import numpy as np

def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()
        return after - before, result
    return wrapper

def generate(size):
    arr = [np.random.uniform(0, 1, size=(size,size)) for _ in range(2)]
    return arr

@timer
def numpy_cos(arr1):
    return np.cos(arr1)

@timer
def c_cos(arr1):
    return c_extension.cos_func_np(arr1)

@timer
def numpy_implementation(arr1, arr2):
    return np.dot(arr1, arr2)

@timer
def c_implementation(arr1, arr2):
    return c_extension.dot_product(arr1.tolist(), arr2.tolist())

@timer
def c_optimized_implementation(arr1, arr2):
    return c_extension.dot_product_optimized(arr1.tolist(), arr2.tolist())

@timer
def c_dot_product_optimized_parallel(arr1, arr2):
    a = arr1.tolist()
    b = arr2.tolist()
    return c_extension.dot_product_optimized_parallel(a, b)

if __name__ == '__main__':
    data = generate(size=10)
    numpy_time_taken, numpy_result = numpy_implementation(*data)
    #c_time_taken, c_result = c_cos(data[0])
    c_time_taken_p, c_result_p = c_dot_product_optimized_parallel(*data)
    #c_optimized_time_taken, c_optimized_result = c_optimized_implementation(*data)
    print(f"time taken with numpy: {numpy_time_taken} seconds")
    #print(f"time taken with optimized c: {c_optimized_time_taken} seconds")
    print(f"time taken with c: {c_time_taken_p} seconds")