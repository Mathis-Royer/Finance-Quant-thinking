"""
Key handling utilities for result dictionaries.

Provides standardized functions for working with result keys
that support both legacy 3-tuple and new 4-tuple formats.
"""

from typing import Tuple, Union

# Type alias for result keys
ResultKey = Union[Tuple[str, str, int], Tuple[str, str, int, str]]


def unpack_key(key: ResultKey) -> Tuple[str, str, int, str]:
    """
    Unpack result key to (strategy, allocation, horizon, config).

    Handles both 3-tuple (legacy) and 4-tuple keys.

    :param key (ResultKey): 3-tuple or 4-tuple key

    :return unpacked (tuple): (strategy, allocation, horizon, config_name)
    """
    if len(key) == 4:
        return key
    return (*key, "baseline")


def make_key(
    strategy: str,
    allocation: str,
    horizon: int,
    config: str = "baseline",
) -> Tuple[str, str, int, str]:
    """
    Create a standardized 4-tuple result key.

    :param strategy (str): Strategy name (E2E, Sup)
    :param allocation (str): Allocation type (Binary, Multi)
    :param horizon (int): Horizon in months
    :param config (str): Config name (baseline, fs, hpt, fs+hpt)

    :return key (tuple): 4-tuple key
    """
    return (strategy, allocation, horizon, config)
