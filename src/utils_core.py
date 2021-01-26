"""
Generic utility functions that are called frequently across modules.
"""
import typing


def default(value: typing.Any, default_value: typing.Any) -> typing.Any:
    """
    Return a default value if a given value is None.
    This is merely a comfort function to avoid typing out "x if x is None else y" over and over again.
    :param value: value that can be None
    :param default_value: default if value is None
    :return: value or default_value
    """
    return default_value if value is None else value


def chunks(lst: typing.List, n: int):
    """
    Yield successive n-sized chunks from lst.
    :param lst: the list to be split.
    :param n: the chunk size.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
