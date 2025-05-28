# radar_framework/radar/AESA/utils/validation.py

import numpy as np

from radar_framework.radar.AESA.exceptions import AESAError


def assert_ndarray(x, name: str, ndim: int):
    """
    Assert that x is a numpy ndarray with the given number of dimensions.
    """
    if not isinstance(x, np.ndarray):
        raise AESAError(f"{name} must be numpy.ndarray, got {type(x).__name__}")
    if x.ndim != ndim:
        raise AESAError(f"{name} must have {ndim} dimensions, got {x.ndim}")


def assert_shape(x, name: str, expected: tuple):
    """
    Assert that ndarray x has the exact expected shape.
    """
    if not isinstance(expected, tuple):
        raise AESAError(f"expected shape for {name} must be a tuple, got {type(expected).__name__}")
    if x.shape != expected:
        raise AESAError(f"{name}.shape must be {expected}, got {x.shape}")


def assert_scalar(x, name: str):
    """
    Assert that x is a scalar (int, float, or complex).
    """
    if not isinstance(x, (int, float, complex)):
        raise AESAError(f"{name} must be scalar (int, float, complex), got {type(x).__name__}")


def assert_positive_scalar(x, name: str):
    """
    Assert that x is a positive scalar.
    """
    assert_scalar(x, name)
    if not x > 0:
        raise AESAError(f"{name} must be positive, got {x}")


def assert_list(x, name: str):
    """
    Assert that x is a list.
    """
    if not isinstance(x, list):
        raise AESAError(f"{name} must be a list, got {type(x).__name__}")


def assert_list_length(x, name: str, expected_len: int):
    """
    Assert that list x has the expected length.
    """
    assert_list(x, name)
    if len(x) != expected_len:
        raise AESAError(f"{name} must have length {expected_len}, got {len(x)}")


def assert_list_of_ndarray(x, name: str, ndim: int):
    """
    Assert that x is a list of numpy ndarrays each with ndim dimensions.
    """
    assert_list(x, name)
    for idx, item in enumerate(x):
        if not isinstance(item, np.ndarray):
            raise AESAError(f"{name}[{idx}] must be numpy.ndarray, got {type(item).__name__}")
        if item.ndim != ndim:
            raise AESAError(f"{name}[{idx}] must have {ndim} dimensions, got {item.ndim}")


def assert_min_length(x, name: str, min_len: int):
    """
    Assert that sequence or ndarray x has at least min_len elements.
    """
    length = len(x) if hasattr(x, '__len__') else None
    if length is None:
        raise AESAError(f"{name} has no length")
    if length < min_len:
        raise AESAError(f"{name} must have at least {min_len} elements, got {length}")


def assert_square_matrix(x, name: str):
    """
    Assert that x is a 2D square numpy ndarray.
    """
    assert_ndarray(x, name, 2)
    if x.shape[0] != x.shape[1]:
        raise AESAError(f"{name} must be square matrix, got shape {x.shape}")
