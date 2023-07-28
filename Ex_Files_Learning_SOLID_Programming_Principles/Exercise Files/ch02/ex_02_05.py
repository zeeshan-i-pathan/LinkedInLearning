"""S.O.L.I.D. Design.

Chapter 2, LSP. Part 5.
"""
from __future__ import annotations

import abc
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev

from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, TextIO, Union


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


RowGenerator = Callable[..., Iterator[dict[str, Any]]]
SampleBuilder = Callable[[dict[str, Any]], Sample]


class Reader:
    def __init__(
        self, row_generator: RowGenerator, sample_builder: SampleBuilder
    ) -> None:
        self.row_generator = row_generator
        self.sample_builder = sample_builder
        self.samples: list[Sample] = []

    def read(self, source: Path) -> None:
        row_gen_class = self.row_generator
        sample_bldr = self.sample_builder
        with source.open() as input:
            row_iter = row_gen_class(input)
            self.samples = [sample_bldr(row) for row in row_iter]


class MDTableRow:
    """Markdown table content transformed into dictionaries."""

    def __init__(self, input: TextIO, table_index: int = 0) -> None:
        self.input = input
        self._index = table_index

    @staticmethod
    def block_parse(text: TextIO) -> list[str]:
        blocks = re.split(r"\n\s*\n", text.read())
        return blocks

    @staticmethod
    def table_block(block: str) -> bool:
        matches = [re.match(r"^(\|.+)+$", line) for line in block.splitlines()]
        return all(matches)

    @staticmethod
    def table_data(block: str) -> Iterator[dict[str, str]]:
        line_iter = iter(block.splitlines())
        line_0 = next(line_iter)
        headings = [col.strip() for col in line_0.split("|")]
        for line in line_iter:
            columns = [col.strip() for col in line.split("|")]
            if all(re.match(r"^-+$", c) is not None for c in columns if c):
                continue
            else:
                yield dict(zip(headings, columns))

    def table_row(self) -> Iterator[dict[str, str]]:
        blocks = self.block_parse(self.input)
        tables = list(filter(self.table_block, blocks))
        first_table = tables[self._index]
        yield from self.table_data(first_table)


class MDReader(Reader):
    """Handle extra parameter to read(), passed to row_generator class"""

    def __init__(
        self, row_generator: RowGenerator, sample_builder: SampleBuilder
    ) -> None:
        super().__init__(row_generator, sample_builder)
        self._index = 0

    def table_index(self, index: int) -> "MDReader":
        self._index = 0
        return self

    def read(self, source: Path) -> None:
        row_gen_class = self.row_generator
        sample_bldr = self.sample_builder
        with source.open() as input:
            row_iter = row_gen_class(input, self._index)
            self.samples = [sample_bldr(row) for row in row_iter]


class Correlation:
    def __init__(self, reader: Reader) -> None:
        self._samples = reader.samples

    def r(self) -> float:
        self._mean_x = mean(s.tach for s in self._samples)
        self._mean_y = mean(s.engine for s in self._samples)
        self._sd_x = stdev(s.tach for s in self._samples)
        self._sd_y = stdev(s.engine for s in self._samples)
        n = len(self._samples)
        r = (
            sum(s.tach * s.engine for s in self._samples)
            - n * self._mean_x * self._mean_y
        ) / ((n - 1) * self._sd_x * self._sd_y)
        return r


def main(source: Path) -> None:
    reader: Reader
    row_generator: RowGenerator
    sample_builder: SampleBuilder
    if source.suffix == ".csv":
        row_generator = csv.DictReader
        sample_builder = lambda row: Sample(
            tach=float(row["Tach"]), engine=float(row["Engine"])
        )
        reader = Reader(row_generator, sample_builder)
        reader.read(source)
    elif source.suffix == ".json":
        row_generator = lambda input: (json.loads(line) for line in input)
        sample_builder = lambda row: Sample(tach=row["Tach"], engine=row["Engine"])
        reader = Reader(row_generator, sample_builder)
        reader.read(source)
    elif source.suffix == ".md":
        row_generator = lambda input, table_index: MDTableRow(
            input, table_index
        ).table_row()
        sample_builder = lambda row: Sample(
            tach=float(row["Tach"]), engine=float(row["Engine"])
        )
        reader = MDReader(row_generator, sample_builder)
        reader.table_index(0).read(source)
    else:
        raise ValueError(f"Can't process {source}")
    analysis = Correlation(reader)
    print(f"Correlation = {analysis.r():.3f}")


test_sample = """
>>> row = {"Tach": "10", "Engine": "8.04"}
>>> s_0 = Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> s_0
Sample(tach=10.0, engine=8.04)
"""

test_csv_reader = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text(
... 'Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68'
... )
>>> row_generator = csv.DictReader
>>> sample_builder = lambda row: Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> reader = Reader(row_generator, sample_builder)
>>> reader.read(data)
>>> len(reader.samples)
11
>>> reader.samples[0]
Sample(tach=10.0, engine=8.04)
>>> reader.samples[-1]
Sample(tach=5.0, engine=5.68)
"""

test_ndjson_reader = """
>>> from pathlib import Path
>>> data = Path("temp.json")
>>> _ = data.write_text(
... '{"Tach": 10.0, "Engine": 9.14}\\n{"Tach": 8.0, "Engine": 8.14}\\n{"Tach": 13.0, "Engine": 8.74}\\n{"Tach": 9.0, "Engine": 8.77}\\n{"Tach": 11.0, "Engine": 9.26}\\n{"Tach": 14.0, "Engine": 8.1}\\n{"Tach": 6.0, "Engine": 6.13}\\n{"Tach": 4.0, "Engine": 3.1}\\n{"Tach": 12.0, "Engine": 9.13}\\n{"Tach": 7.0, "Engine": 7.26}\\n{"Tach": 5.0, "Engine": 4.74}\\n'
... )
>>> row_generator = lambda input: (json.loads(line) for line in input)
>>> sample_builder = lambda row: Sample(tach=row["Tach"], engine=row["Engine"])
>>> reader = Reader(row_generator, sample_builder)
>>> reader.read(data)
>>> len(reader.samples)
11
>>> reader.samples[0]
Sample(tach=10.0, engine=9.14)
>>> reader.samples[-1]
Sample(tach=5.0, engine=4.74)
"""

test_md_reader = """
>>> from pathlib import Path
>>> data = Path("temp.md")
>>> _ = data.write_text(
... '| Sample\\t| Tach\\t| Engine\\n| --------- | ----  | ------\\n| 1\\t| 1000\\t|  883\\n| 2\\t| 1500\\t| 1242\\n| 3\\t| 1500\\t| 1217\\n| 4\\t| 1600\\t| 1306\\n| 5\\t| 1750\\t| 1534\\n| 6\\t| 2000\\t| 1805\\n| 7\\t| 2000\\t| 1720'
... )

>>> row_generator = lambda input, table_index: MDTableRow(input, table_index).table_row()
>>> sample_builder = lambda row: Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> reader = MDReader(row_generator, sample_builder)
>>> reader.table_index(0).read(data)
>>> len(reader.samples)
7
>>> reader.samples[0]
Sample(tach=1000.0, engine=883.0)
>>> reader.samples[-1]
Sample(tach=2000.0, engine=1720.0)

"""

test_correlation = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> row_generator = csv.DictReader
>>> sample_builder = lambda row: Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> reader = Reader(row_generator, sample_builder)
>>> reader.read(data)
>>> analysis = Correlation(reader)
>>> f"{analysis.r()=:.3f}"
'analysis.r()=0.816'
>>> f"{analysis._mean_x:.2f}"
'9.00'
>>> f"{analysis._mean_y:.2f}"
'7.50'
>>> f"{analysis._sd_x:.3f}"
'3.317'
>>> f"{analysis._sd_y:.3f}"
'2.032'
"""

test_main = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> main(data)
Correlation = 0.816
"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)

    # main(Path("temp.md"))
