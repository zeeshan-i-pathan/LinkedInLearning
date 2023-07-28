"""S.O.L.I.D. Design.

Chapter 1, ISP. Test Cases.
"""
from unittest.mock import Mock

import pytest

from ex_01_01 import Correlation


@pytest.fixture
def mock_reader() -> Mock:
    reader = Mock(
        samples=[
            Mock(tach=10.0, engine=8.04),
            Mock(tach=8.0, engine=6.95),
            Mock(tach=13.0, engine=7.58),
            Mock(tach=9.0, engine=8.81),
            Mock(tach=11.0, engine=8.33),
            Mock(tach=14.0, engine=9.96),
            Mock(tach=6.0, engine=7.24),
            Mock(tach=4.0, engine=4.26),
            Mock(tach=12.0, engine=10.84),
            Mock(tach=7.0, engine=4.82),
            Mock(tach=5.0, engine=5.68),
        ]
    )
    return reader


def test_correlation(mock_reader: Mock) -> None:
    c = Correlation(mock_reader)
    assert c.r() == pytest.approx(0.816, abs=1e-3)
