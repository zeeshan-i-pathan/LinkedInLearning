"""S.O.L.I.D. Design.

Chapter 5, SRP. Test Cases.
"""
from pathlib import Path
import pytest
from unittest.mock import Mock, sentinel, call
import ex_05_01


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


def test_reads_samples(
    mock_row_gen: Mock, mock_sample_bld: Mock, mock_source_path: Path
) -> None:
    """
    SCENARIO: Reads Samples and creates a SampleList
      GIVEN a row_generator and a sample_builder
      AND a Reader built from the row_generator and the sample_builder
      AND a source of data
      WHEN data is read from the source
      THEN the row_generator is invoked once
      AND the sample_builder is invoked once for each row of data
    """
    rdr = ex_05_01.Reader(mock_row_gen, mock_sample_bld)
    rdr.read(mock_source_path)
    assert len(mock_row_gen.mock_calls) == 1
    assert mock_sample_bld.mock_calls == [call(sentinel.data)]
    assert rdr.samples == [sentinel.sample]
