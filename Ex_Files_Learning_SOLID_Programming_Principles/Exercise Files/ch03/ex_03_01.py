"""S.O.L.I.D. Design.

Chapter 3, OCP. Part 1.
"""
from __future__ import annotations

import csv
from functools import cache
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, TextIO, Union


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


class SampleList(list[Sample]):
    def sum(self, attribute: str) -> float:
        return sum(getattr(item, attribute) for item in self)

    def n(self, attribute: str) -> float:
        return sum(1 for item in self)

    def sum_2(self, attribute: str) -> float:
        return sum(getattr(item, attribute) ** 2 for item in self)

    def mean(self, attribute: str) -> float:
        return self.sum(attribute) / self.n(attribute)

    def stdev(self, attribute: str) -> float:
        n = self.n(attribute)
        mean_2 = (self.sum(attribute) / n) ** 2
        squares = self.sum_2(attribute) / (n - 1)
        return sqrt(squares - mean_2 * n / (n - 1))


class Correlation:
    def __init__(self, samples: SampleList) -> None:
        self.samples = samples

    def r(self, attr_1: str, attr_2: str) -> float:
        n = self.samples.n(attr_1)
        r = (
            sum(getattr(s, attr_1) * getattr(s, attr_2) for s in self.samples)
            - n * self.samples.mean(attr_1) * self.samples.mean(attr_2)
        ) / ((n - 1) * self.samples.stdev(attr_1) * self.samples.stdev(attr_2))
        return r


class Regression(Correlation):
    def __init__(self, samples: SampleList) -> None:
        super().__init__(samples)
        self._a: float
        self._b: float

    def compute_long(self, attr_1: str, attr_2: str) -> None:
        if not hasattr(self, "_a") or not hasattr(self, "_b"):
            n = self.samples.n(attr_1)
            self._b = (
                sum(getattr(s, attr_1) * getattr(s, attr_2) for s in self.samples)
                - n * self.samples.mean(attr_1) * self.samples.mean(attr_2)
            ) / ((n - 1) * self.samples.stdev(attr_1) * self.samples.stdev(attr_1))
            self._a = self.samples.mean(attr_2) - self._b * self.samples.mean(attr_1)

    def compute(self, attr_1: str, attr_2: str) -> None:
        if not (hasattr(self, "_a") and hasattr(self, "_b")):
            self._b = (
                self.r(attr_1, attr_2)
                * self.samples.stdev(attr_2)
                / self.samples.stdev(attr_1)
            )
            self._a = self.samples.mean(attr_2) - self._b * self.samples.mean(attr_1)

    def b(self, attr_1: str, attr_2: str) -> float:
        self.compute(attr_1, attr_2)
        return self._b

    def a(self, attr_1: str, attr_2: str) -> float:
        self.compute(attr_1, attr_2)
        return self._a


RowGenerator = Callable[..., Iterator[dict[str, Any]]]
SampleBuilder = Callable[[dict[str, Any]], Sample]


class Reader:
    sample_collection_class = SampleList

    def __init__(
        self, row_generator: RowGenerator, sample_builder: SampleBuilder
    ) -> None:
        self.row_generator = row_generator
        self.sample_builder = sample_builder
        self.samples = self.sample_collection_class()

    def read(self, source: Path) -> None:
        row_gen_class = self.row_generator
        sample_bldr = self.sample_builder
        with source.open() as input:
            row_iter = row_gen_class(input)
            self.samples = SampleList(sample_bldr(row) for row in row_iter)


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

test_sample_list = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text(
... 'Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68'
... )
>>> row_generator = csv.DictReader
>>> sample_builder = lambda row: Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> reader = Reader(row_generator, sample_builder)
>>> reader.read(data)
>>> f"{reader.samples.mean('tach')=:.2f}"
"reader.samples.mean('tach')=9.00"
>>> f"{reader.samples.stdev('tach')=:.3f}"
"reader.samples.stdev('tach')=3.317"
>>> round(stdev(s.tach for s in reader.samples), 3)
3.317
>>> f"{reader.samples.mean('engine')=:.2f}"
"reader.samples.mean('engine')=7.50"
>>> f"{reader.samples.stdev('engine')=:.3f}"
"reader.samples.stdev('engine')=2.032"
>>> round(stdev(s.engine for s in reader.samples), 3)
2.032
>>> analysis = Regression(reader.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"
>>> f"y = {analysis.b('tach', 'engine'):.3f} * x + {analysis.a('tach', 'engine'):.3f}"
'y = 0.500 * x + 3.000'
"""

test_correlation = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> row_generator = csv.DictReader
>>> sample_builder = lambda row: Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))
>>> reader = Reader(row_generator, sample_builder)
>>> reader.read(data)
>>> analysis = Correlation(reader.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"
"""


__test__ = {name: value for name, value in locals().items() if name.startswith("test")}


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
