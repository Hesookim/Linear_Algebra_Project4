import time


def benchmark(function, *args):
    """
    Measure execution time of a function.
    """

    start = time.perf_counter()

    result = function(*args)

    end = time.perf_counter()

    elapsed = end - start

    return result, elapsed