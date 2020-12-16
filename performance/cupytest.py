import time
import numpy as np
import cupy as cp


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        result = func(*args, **kwargs)
        after = time.time()
        return after - before, result

    return wrapper


def generate(asize, lsize):
    return [np.random.uniform(0, 1, size=(asize, asize)) for i in range(lsize)]


@timer
def np_test_dot(a, b):
    res = []
    for i in range(len(a)):
        res.append(np.dot(a[0], b[0]))
    return res


@timer
def cp_test_dot(a, b):
    res = []
    for i in range(len(a)):
        res.append(cp.dot(a[0], b[0]))
    return res


if __name__ == '__main__':
    t_np = 0
    t_cp = 0
    N = 10
    for i in range(N):
        data1 = generate(1000, 10)
        data2 = generate(1000, 10)
        t_np_dot, r_np_dot = np_test_dot(data1, data2)
        t_cp_dot, r_cp_dot = cp_test_dot(data1, data2)
        t_np += t_np_dot
        t_cp += t_cp_dot
    print("Time with numpy :", t_np / N)
    print('Time with cupy :', t_cp / N)
