"""Tests for robot_control.scripts.cli.input_helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robot_control.scripts.cli.input_helpers import ask_float, ask_int


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mock_text_sequence(values: list[str | None]) -> MagicMock:
    """Return a side-effect for ``questionary.text()`` that yields *values*.

    Each element becomes the return value of ``.ask()``.  A ``None``
    element simulates the user pressing Ctrl-C.
    """
    calls = iter(values)

    def _factory(*args, **kwargs):  # noqa: ANN
        mock_question = MagicMock()
        mock_question.ask.return_value = next(calls)
        return mock_question

    return _factory


# ------------------------------------------------------------------
# ask_float
# ------------------------------------------------------------------

class TestAskFloat:
    """Tests for ``ask_float``."""

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_valid_input(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence(["3.14"])
        result = ask_float("Enter value", default=1.0)
        assert result == pytest.approx(3.14)

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_empty_returns_default(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence([""])
        result = ask_float("Enter value", default=2.5)
        assert result == pytest.approx(2.5)

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_cancel_raises_keyboard_interrupt(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence([None])
        with pytest.raises(KeyboardInterrupt):
            ask_float("Enter value", default=0.0)

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_out_of_range_low_retries(self, mock_q: MagicMock) -> None:
        """First attempt is below min_val, second is valid."""
        mock_q.text.side_effect = _mock_text_sequence(["-5", "3.0"])
        result = ask_float("Enter value", default=1.0, min_val=0.0)
        assert result == pytest.approx(3.0)
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_out_of_range_high_retries(self, mock_q: MagicMock) -> None:
        """First attempt exceeds max_val, second is valid."""
        mock_q.text.side_effect = _mock_text_sequence(["100", "5.0"])
        result = ask_float("Enter value", default=1.0, max_val=10.0)
        assert result == pytest.approx(5.0)
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_non_numeric_retries(self, mock_q: MagicMock) -> None:
        """Non-numeric text triggers a retry."""
        mock_q.text.side_effect = _mock_text_sequence(["abc", "7.5"])
        result = ask_float("Enter value", default=1.0)
        assert result == pytest.approx(7.5)
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_min_and_max_bounds(self, mock_q: MagicMock) -> None:
        """Value exactly at bounds is accepted."""
        mock_q.text.side_effect = _mock_text_sequence(["0.0"])
        assert ask_float("V", default=5.0, min_val=0.0, max_val=10.0) == 0.0

        mock_q.text.side_effect = _mock_text_sequence(["10.0"])
        assert ask_float("V", default=5.0, min_val=0.0, max_val=10.0) == 10.0


# ------------------------------------------------------------------
# ask_int
# ------------------------------------------------------------------

class TestAskInt:
    """Tests for ``ask_int``."""

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_valid_input(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence(["42"])
        assert ask_int("Count", default=1) == 42

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_empty_returns_default(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence([""])
        assert ask_int("Count", default=10) == 10

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_cancel_raises_keyboard_interrupt(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence([None])
        with pytest.raises(KeyboardInterrupt):
            ask_int("Count", default=0)

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_out_of_range_low_retries(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence(["-1", "3"])
        assert ask_int("Count", default=1, min_val=0) == 3
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_out_of_range_high_retries(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence(["20", "5"])
        assert ask_int("Count", default=1, max_val=10) == 5
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_non_integer_retries(self, mock_q: MagicMock) -> None:
        """Float string or text triggers a retry."""
        mock_q.text.side_effect = _mock_text_sequence(["3.14", "3"])
        assert ask_int("Count", default=1) == 3
        assert mock_q.text.call_count == 2

    @patch("robot_control.scripts.cli.input_helpers.questionary")
    def test_boundary_values_accepted(self, mock_q: MagicMock) -> None:
        mock_q.text.side_effect = _mock_text_sequence(["0"])
        assert ask_int("V", default=5, min_val=0, max_val=10) == 0

        mock_q.text.side_effect = _mock_text_sequence(["10"])
        assert ask_int("V", default=5, min_val=0, max_val=10) == 10
