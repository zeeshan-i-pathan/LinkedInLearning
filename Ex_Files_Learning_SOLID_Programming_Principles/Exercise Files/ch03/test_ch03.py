"""S.O.L.I.D. Design.

Chapter 3, OCP. Test Cases.
"""
import logging
from pathlib import Path
from unittest.mock import Mock, call, sentinel

import pytest

import ex_03_04
import ex_03_05
import ex_03_06
import toml  # type: ignore [import]


def test_main_logging(
    caplog: pytest.LogCaptureFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "temp.csv"
    data.write_text(
        "Tach,Engine\n10.0,8.04\n8.0,6.95\n13.0,7.58\n9.0,8.81\n11.0,8.33\n14.0,9.96\n6.0,7.24\n4.0,4.26\n12.0,10.84\n7.0,4.82\n5.0,5.68"
    )
    caplog.set_level(logging.DEBUG)
    ex_03_05.main(data)
    out, err = capsys.readouterr()
    assert out == "Correlation = 0.816\n"
    assert caplog.record_tuples == [
        ("Reader", 10, f"read(*({repr(data)},), **{{}}) = None")
    ]


@pytest.fixture
def mock_row_gen() -> Mock:
    return Mock(name="Row Generator", return_value=[sentinel.data])


@pytest.fixture
def mock_sample_bld() -> Mock:
    return Mock(name="Sample Builder", return_value=sentinel.sample)


@pytest.fixture
def mock_source_path(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.write_text("data\n")
    return data


def test_reader_mocked(
    mock_row_gen: Mock, mock_sample_bld: Mock, mock_source_path: Path
) -> None:
    """GIVEN a mock row_generator, a mock sample_builder, and a mock source path
    AND a Reader built from the mock row_generator, and the mock sample_builder
    WHEN data is read from the temporary path
    THEN the mock row_generator is invoked once
    AND the mock sample_builder is invoked once for each piece of mocked data
    """
    rdr = ex_03_06.Reader(mock_row_gen, mock_sample_bld)
    rdr.read(mock_source_path)
    assert len(mock_row_gen.mock_calls) == 1
    assert mock_sample_bld.mock_calls == [call(sentinel.data)]


def test_main_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "temp.csv"
    data.write_text(
        "Tach,Engine\n10.0,8.04\n8.0,6.95\n13.0,7.58\n9.0,8.81\n11.0,8.33\n14.0,9.96\n6.0,7.24\n4.0,4.26\n12.0,10.84\n7.0,4.82\n5.0,5.68"
    )
    ex_03_04.main(data)
    out, err = capsys.readouterr()
    assert out.splitlines() == ["Correlation = 0.816"]
