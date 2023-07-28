"""S.O.L.I.D. Design.

Chapter 3, OCP. Part 6.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from functools import cache
from math import sqrt
from operator import attrgetter
from pathlib import Path
from statistics import mean, stdev

import toml  # type: ignore [import]
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TextIO,
    Union,
)


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


class StatsMethods(Collection[Any]):
    def sum(self, attribute: str) -> float:
        return sum(map(attrgetter(attribute), self))

    def n(self, attribute: str) -> float:
        return sum(1 for item in self)

    def sum_2(self, attribute: str) -> float:
        # return sum(getattr(item, attribute) ** 2 for item in self)
        v: Iterator[float] = map(attrgetter(attribute), self)
        v_2 = map(lambda x: x ** 2, v)
        return sum(v_2)

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


class SampleList(list[Sample], StatsMethods):
    pass


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
            self.samples = self.sample_collection_class(
                sample_bldr(row) for row in row_iter
            )


class Configuration:
    READER = Reader
    FORMATS: dict[str, dict[str, Any]] = {
        "csv": dict(
            row_generator=csv.DictReader,
            sample_builder=lambda row: Sample(
                tach=float(row["Tach"]), engine=float(row["Engine"])
            ),
        ),
        "ndjson": dict(
            row_generator=lambda input: (json.loads(line) for line in input),
            sample_builder=lambda row: Sample(tach=row["Tach"], engine=row["Engine"]),
        ),
        # toml
        # markdown
    }


def get_options(argv: list[str] = sys.argv[1:]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        default=Path(os.environ.get("SOURCE") or "-"),
    )
    parser.add_argument("-f", "--format", choices=["csv", "ndjson", "toml", "md"])
    parser.add_argument("-t", "--table", type=int, help="MD table number")
    options = parser.parse_args(argv)
    if options.format is None:
        SUFFIX_MAP = {".csv": "csv", ".json": "ndjson", ".toml": "toml", ".md": "md"}
        try:
            options.format = SUFFIX_MAP[options.source.suffix]
        except KeyError:
            parser.error(f"Unknown Suffix on {options.source}")
    return options


def main() -> None:
    options = get_options()
    config = Configuration.FORMATS[options.format]
    reader = Configuration.READER(config["row_generator"], config["sample_builder"])
    reader.read(options.source)
    analysis = Correlation(reader.samples)
    print(f"{analysis.r('tach', 'engine')=:.3f}")


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
>>> analysis = Correlation(reader.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"
"""

test_config_1 = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text(
... 'Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68'
... )
>>> config_path = Path("APP.toml")
>>> with config_path.open("w") as config_toml:
...     _ = toml.dump({"APP": {"description": "sample config"}}, config_toml)
>>> options = get_options([str(data)])
>>> options
Namespace(source=PosixPath('temp.csv'), format='csv', table=None)
>>> config = Configuration.FORMATS[options.format]
>>> reader = Reader(config['row_generator'], config['sample_builder'])
>>> reader.read(options.source)
>>> len(reader.samples)
11
>>> analysis = Correlation(reader.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"

"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}

if __name__ == "__main__":
    # import doctest
    #
    # doctest.testmod(verbose=True)
    main()
