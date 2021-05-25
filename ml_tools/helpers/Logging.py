import logging


class Logging:
    def logging_output(parameter):
        def decorator(func):
            def wrapper(*args):
                print(args)
                print('Fitting {} begins...'.format(parameter))
                func(*args)
                print('Fitting {} ends...'.format(parameter))
            return wrapper
        return decorator
