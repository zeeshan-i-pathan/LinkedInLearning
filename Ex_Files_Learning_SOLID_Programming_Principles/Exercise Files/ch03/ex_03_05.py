"""S.O.L.I.D. Design.

Chapter 3, OCP. Part 5.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
from functools import wraps
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Type, cast


def logged(class_: Type[Any]) -> Type[Any]:
    class_.logger = logging.getLogger(class_.__qualname__)
    return class_


def trace(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kw: Any) -> Any:
        result = method(self, *args, **kw)
        self.logger.debug(f"{method.__name__}(*{args!r}, **{kw!r}) = {result!r}")
        return result

    return wrapped


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


class SampleList(List[Sample]):
    mean_function = staticmethod(statistics.mean)
    stdev_function = staticmethod(statistics.stdev)

    def n(self, attribute: str) -> float:
        return sum(1 for _ in self)

    def mean(self, attribute: str) -> float:
        return cast(float, self.mean_function(getattr(s, attribute) for s in self))

    def stdev(self, attribute: str) -> float:
        return cast(float, self.stdev_function(getattr(s, attribute) for s in self))


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


RowGenerator = Callable[..., Iterator[dict[str, Any]]]
SampleBuilder = Callable[[dict[str, Any]], Sample]


@logged
class Reader:
    sample_collection_class = SampleList

    def __init__(
        self, row_generator: RowGenerator, sample_builder: SampleBuilder
    ) -> None:
        self.row_generator = row_generator
        self.sample_builder = sample_builder
        self.samples = self.sample_collection_class()

    @trace
    def read(self, source: Path) -> None:
        row_gen_class = self.row_generator
        sample_bldr = self.sample_builder
        with source.open() as input:
            try:
                row_iter = row_gen_class(input)
            except ValueError:
                self.logger.error("Could not process {source}")  # type: ignore [attr-defined]

            self.samples = self.sample_collection_class(
                sample_bldr(row) for row in row_iter
            )


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
    elif source.suffix == ".json":
        row_generator = lambda input: (json.loads(line) for line in input)
        sample_builder = lambda row: Sample(tach=row["Tach"], engine=row["Engine"])
        reader = Reader(row_generator, sample_builder)
    else:
        raise ValueError(f"Can't process {source}")
    reader.read(source)

    analysis = Correlation(reader.samples)
    print(f"Correlation = {analysis.r('tach', 'engine'):.3f}")


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
>>> round(statistics.stdev(s.tach for s in reader.samples), 3)
3.317
>>> f"{reader.samples.mean('engine')=:.2f}"
"reader.samples.mean('engine')=7.50"
>>> f"{reader.samples.stdev('engine')=:.3f}"
"reader.samples.stdev('engine')=2.032"
>>> round(statistics.stdev(s.engine for s in reader.samples), 3)
2.032
>>> analysis = Correlation(reader.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"
"""

test_main = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> main(data)
Correlation = 0.816
>>> logging.shutdown()
"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
