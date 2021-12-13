import time

# helper
def tell_us_youre_running(func):
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}")
        start_time = time.time()
        res = func(*args, **kwargs)
        print(f"{func.__name__} ran in {time.time() - start_time} seconds")
        return res

    wrapper.__name__ = func.__name__

    return wrapper


def tell_us_return_size(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        print(f"{func.__name__} returned {len(res)} items")
        return res

    wrapper.__name__ = func.__name__

    return wrapper
