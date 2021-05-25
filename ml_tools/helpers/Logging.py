import logging
from functools import wraps


class Logging:
    def logging_output(parameter):
        def decorator(func):
            @wraps(func)
            def wrapper(*args):
                print('Fitting {} begins...'.format(parameter))
                return func(*args)
            return wrapper
        return decorator
