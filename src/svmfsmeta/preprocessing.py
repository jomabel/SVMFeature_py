"""Preprocessing used by the repository examples and dashboard."""


def minmax_scale(data, inputs):
    """Return a copy with predictors scaled to [0, 1], constants to 0.5.

    This follows the R examples. The response and unused columns are unchanged;
    the optimizer itself does not normalize data automatically.
    """
    inputs = list(inputs)
    values = data[inputs].astype(float)
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    scaled = (values - minimum) / span.mask(span == 0, 1)
    scaled.loc[:, span == 0] = 0.5
    result = data.copy()
    result[inputs] = scaled
    return result
