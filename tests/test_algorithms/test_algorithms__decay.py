"""Unit tests for algorithms/decay/"""

# Authors: Genevieve Hayes (modified by Kyle Nakamura)
# License: BSD 3-clause

import pytest

from mlrose_ky import GeomDecay, ArithDecay, ExpDecay, CustomSchedule


class TestAlgorithmsDecay:
    """Test cases for the algorithms.decay module."""

    # Testing GeomDecay
    def test_geom_decay_evaluate_above_min(self):
        """Test GeomDecay evaluate method when result is above min_temp."""
        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        assert schedule.evaluate(5) == 7.737809374999998

    def test_geom_decay_evaluate_below_min(self):
        """Test GeomDecay evaluate method when result is below min_temp."""
        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        assert schedule.evaluate(50) == 1

    def test_geom_decay_init_invalid_parameters(self):
        """Test GeomDecay initialization with invalid parameters."""
        with pytest.raises(ValueError):
            GeomDecay(init_temp=0)
        with pytest.raises(ValueError):
            GeomDecay(init_temp=10, decay=0)
        with pytest.raises(ValueError):
            GeomDecay(init_temp=10, decay=1.1)
        with pytest.raises(ValueError):
            GeomDecay(init_temp=10, min_temp=0)
        with pytest.raises(ValueError):
            GeomDecay(init_temp=10, min_temp=10)

    def test_geom_decay_str_repr(self):
        """Test GeomDecay __str__ and __repr__ methods."""
        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        assert str(schedule) == "GeomDecay(init_temp=10, decay=0.95, min_temp=1)"
        assert repr(schedule) == "GeomDecay(init_temp=10, decay=0.95, min_temp=1)"

    def test_geom_decay_eq(self):
        """Test GeomDecay __eq__ method."""
        schedule1 = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        schedule2 = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        schedule3 = GeomDecay(init_temp=5, decay=0.95, min_temp=1)
        assert schedule1 == schedule2
        assert schedule1 != schedule3
        assert schedule1 != "NotASchedule"

    def test_geom_decay_get_info(self):
        """Test GeomDecay get_info__ method."""
        schedule = GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        info = schedule.get_info__(t=5, prefix="test")
        expected_info = {
            "test__schedule_current_value": 7.737809374999998,
            "test__schedule_decay": 0.95,
            "test__schedule_init_temp": 10,
            "test__schedule_min_temp": 1,
            "test__schedule_type": "geometric",
        }
        assert info == expected_info

    # Testing ArithDecay
    def test_arith_decay_evaluate_above_min(self):
        """Test ArithDecay evaluate method when result is above min_temp."""
        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        assert schedule.evaluate(5) == 5.25

    def test_arith_decay_evaluate_below_min(self):
        """Test ArithDecay evaluate method when result is below min_temp."""
        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        assert schedule.evaluate(50) == 1

    def test_arith_decay_init_invalid_parameters(self):
        """Test ArithDecay initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ArithDecay(init_temp=0)
        with pytest.raises(ValueError):
            ArithDecay(init_temp=10, decay=0)
        with pytest.raises(ValueError):
            ArithDecay(init_temp=10, decay=1.1)
        with pytest.raises(ValueError):
            ArithDecay(init_temp=10, min_temp=0)
        with pytest.raises(ValueError):
            ArithDecay(init_temp=10, min_temp=10)

    def test_arith_decay_str_repr(self):
        """Test ArithDecay __str__ and __repr__ methods."""
        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        expected_str = "ArithDecay(init_temp=10, decay=0.95, min_temp=1)"
        assert str(schedule) == expected_str
        assert repr(schedule) == expected_str

    def test_arith_decay_eq(self):
        """Test ArithDecay __eq__ method."""
        schedule1 = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        schedule2 = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        schedule3 = ArithDecay(init_temp=5, decay=0.95, min_temp=1)
        assert schedule1 == schedule2
        assert schedule1 != schedule3
        assert schedule1 != "NotASchedule"

    def test_arith_decay_get_info(self):
        """Test ArithDecay get_info__ method."""
        schedule = ArithDecay(init_temp=10, decay=0.95, min_temp=1)
        info = schedule.get_info__(t=5, prefix="test_")
        expected_info = {
            "test_schedule_type": "arithmetic",
            "test_schedule_init_temp": 10,
            "test_schedule_decay": 0.95,
            "test_schedule_min_temp": 1,
            "test_schedule_current_value": 5.25,
        }
        assert info == expected_info

    # Testing ExpDecay
    def test_exp_decay_evaluate_above_min(self):
        """Test ExpDecay evaluate method when result is above min_temp."""
        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        assert schedule.evaluate(5) == 7.788007830714049

    def test_exp_decay_evaluate_below_min(self):
        """Test ExpDecay evaluate method when result is below min_temp."""
        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        assert schedule.evaluate(50) == 1.0

    def test_exp_decay_init_invalid_parameters(self):
        """Test ExpDecay initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ExpDecay(init_temp=0)
        with pytest.raises(ValueError):
            ExpDecay(init_temp=10, exp_const=0)
        with pytest.raises(ValueError):
            ExpDecay(init_temp=10, exp_const=-0.1)
        with pytest.raises(ValueError):
            ExpDecay(init_temp=10, min_temp=0)
        with pytest.raises(ValueError):
            ExpDecay(init_temp=10, min_temp=10)

    def test_exp_decay_str_repr(self):
        """Test ExpDecay __str__ and __repr__ methods."""
        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        expected_str = "ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)"
        assert str(schedule) == expected_str
        assert repr(schedule) == expected_str

    def test_exp_decay_eq(self):
        """Test ExpDecay __eq__ method."""
        schedule1 = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        schedule2 = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        schedule3 = ExpDecay(init_temp=5, exp_const=0.05, min_temp=1)
        assert schedule1 == schedule2
        assert schedule1 != schedule3
        assert schedule1 != "NotASchedule"

    def test_exp_decay_get_info(self):
        """Test ExpDecay get_info__ method."""
        schedule = ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
        info = schedule.get_info__(t=5, prefix="test_")
        expected_info = {
            "test_schedule_type": "exponential",
            "test_schedule_init_temp": 10,
            "test_schedule_exp_const": 0.05,
            "test_schedule_min_temp": 1,
            "test_schedule_current_value": 7.788007830714049,
        }
        assert info == expected_info

    # Testing CustomSchedule
    def test_custom_schedule_evaluate(self):
        """Test CustomSchedule evaluate method."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_decay_function(time: int, offset: int) -> float:
            return time + offset

        kwargs = {"offset": 10}
        schedule = CustomSchedule(custom_decay_function, **kwargs)
        assert schedule.evaluate(5) == 15

    def test_custom_schedule_init_invalid_function(self):
        """Test CustomSchedule initialization with invalid callback function."""
        with pytest.raises(TypeError, match="'schedule' must be a function"):

            # noinspection PyMissingOrEmptyDocstring
            class NotCallable:
                pass

            # noinspection PyTypeChecker
            CustomSchedule(NotCallable)

    def test_custom_schedule_str_repr(self):
        """Test CustomSchedule __str__ and __repr__ methods."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_decay_function(time: int) -> float:
            return time

        schedule = CustomSchedule(custom_decay_function)
        expected_str = "CustomSchedule(schedule=custom_decay_function, kwargs={})"
        assert str(schedule) == expected_str
        assert repr(schedule) == expected_str

    def test_custom_schedule_eq(self):
        """Test CustomSchedule __eq__ method."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_decay_function1(time: int) -> float:
            return time

        # noinspection PyMissingOrEmptyDocstring
        def custom_decay_function2(time: int) -> float:
            return time * 2

        schedule1 = CustomSchedule(custom_decay_function1)
        schedule2 = CustomSchedule(custom_decay_function1)
        schedule3 = CustomSchedule(custom_decay_function2)
        assert schedule1 == schedule2
        assert schedule1 != schedule3
        assert schedule1 != "NotASchedule"

    def test_custom_schedule_get_info(self):
        """Test CustomSchedule get_info__ method."""

        # noinspection PyMissingOrEmptyDocstring
        def custom_decay_function(time: int, offset: int) -> float:
            return time + offset

        kwargs = {"offset": 10}
        schedule = CustomSchedule(custom_decay_function, **kwargs)
        info = schedule.get_info__(t=5, prefix="test_")
        expected_info = {
            "test_schedule_type": "custom",
            "test_schedule_schedule": "custom_decay_function",
            "test_schedule_param_offset": 10,
            "test_schedule_current_value": 15,
        }
        assert info == expected_info
