"""Unit tests for algorithms/decorators"""

# Author: Kyle Nakamura
# License: BSD 3-clause

from mlrose_ky.decorators import short_name, get_short_name


def test_short_name_assignment():
    """Test that the short_name decorator assigns the correct short name to a function."""

    # noinspection PyMissingOrEmptyDocstring
    @short_name("SomeFunc")
    def some_function():
        return True

    assert hasattr(some_function, "__short_name__"), "The function should have a '__short_name__' attribute"
    assert some_function.__short_name__ == "SomeFunc", "The short name should be 'SomeFunc'"


def test_get_short_name_with_assigned_name():
    """Test retrieving the short name when it has been assigned."""

    # noinspection PyMissingOrEmptyDocstring
    @short_name("SomeFunc")
    def some_function():
        return True

    assert get_short_name(some_function) == "SomeFunc", "The short name should be 'SomeFunc'"


def test_get_short_name_without_assigned_name():
    """Test that the default function name is returned when no short name is assigned."""

    # noinspection PyMissingOrEmptyDocstring
    def some_function():
        return False

    assert get_short_name(some_function) == "some_function", "Should return the default function name 'some_function'"
