from sklearn.metrics import mean_squared_error


def negative_mean_squared_error(*args, **kwargs):
    return -1 * mean_squared_error(*args, **kwargs)
