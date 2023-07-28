"""S.O.L.I.D. Design.

Chapter 5, SRP. Part 1.
"""
from __future__ import annotations
import abc
import argparse
import csv
from dataclasses import dataclass
from functools import cache
import json
from math import sqrt
from operator import attrgetter
import os
from pathlib import Path
import re
from statistics import mean, stdev
import sys
import toml  # type: ignore [import]
from typing import (
    Optional,
    Iterable,
    Iterator,
    TextIO,
    Callable,
    Any,
    Union,
    # List,
    Collection,
    Hashable,
    Sequence,
    Type,
    Protocol,
    NamedTuple,
)


@dataclass(frozen=True)
class Sample:
    """A single measurement.

    >>> s = Sample(tach=1000., engine=883.)
    >>> s
    Sample(tach=1000.0, engine=883.0)
    >>> s.tach
    1000.0
    >>> s.engine
    883.0
    """

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


class Regression(Correlation):
    def __init__(self, samples: SampleList) -> None:
        super().__init__(samples)
        self._a: float
        self._b: float

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


class SampleList(list[Sample], StatsMethods):
    pass


class RowGenerator(Protocol):
    def __init__(self, source: TextIO, **kwargs: Any) -> None:
        ...

    def __iter__(self) -> Iterator[dict[str, Any]]:
        ...


SampleBuilder = Callable[[dict[str, Any]], Sample]


class ReaderBase(abc.ABC):
    """Is this really necessary? Probably not."""

    sample_collection_class = SampleList

    @abc.abstractmethod
    def __init__(
        self,
        row_generator: Type[RowGenerator],
        sample_builder: SampleBuilder,
        **kwargs: Any,
    ) -> None:
        ...

    @abc.abstractmethod
    def read(self, source: Path) -> None:
        ...


class Reader(ReaderBase):
    sample_collection_class = SampleList

    def __init__(
        self,
        row_generator: Type[RowGenerator],
        sample_builder: SampleBuilder,
        **kwargs: Any,
    ) -> None:
        self.row_generator = row_generator
        self.sample_builder = sample_builder
        self.additional_kwargs = kwargs
        self.samples = self.sample_collection_class()

    def read(self, source: Path) -> None:
        row_gen_class = self.row_generator
        sample_bldr = self.sample_builder
        with source.open() as input:
            row_iter = row_gen_class(input, **self.additional_kwargs)
            self.samples = self.sample_collection_class(
                sample_bldr(row) for row in row_iter
            )


class NDJSONRow:
    def __init__(self, source: TextIO):
        self.source = source

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (json.loads(line) for line in self.source)


class MDTableRow:
    """Markdown table content transformed into dictionaries."""

    def __init__(self, input: TextIO, table_index: int = 0) -> None:
        self.input = input
        self.table_index = table_index

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

    def __iter__(self) -> Iterator[dict[str, str]]:
        blocks = self.block_parse(self.input)
        tables = list(filter(self.table_block, blocks))
        first_table = tables[self.table_index]
        yield from self.table_data(first_table)


class TOMLRow:
    def __init__(self, source: TextIO):
        self.data: list[dict[str, Any]] = toml.loads(source).get("APP")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.data)


def build_reader(config: Type[Configuration], options: argparse.Namespace) -> Reader:
    row_generator = config.FORMATS[options.format].row_generator
    sample_builder = config.FORMATS[options.format].sample_builder
    additional = {
        name: getattr(options, name, None)
        for name in config.FORMATS[options.format].additional
    }
    return Reader(row_generator, sample_builder, **additional)


class Analysis(abc.ABC):
    def load(self, config: Type[Configuration], options: argparse.Namespace) -> None:
        self.reader = build_reader(config, options)
        self.reader.read(options.source)

    @abc.abstractmethod
    def report(self, options: argparse.Namespace) -> None:
        pass

    def run(self, config: Type[Configuration], argv: list[str] = sys.argv[1:]) -> None:
        options = get_options(argv)
        self.load(config, options)
        self.report(options)


class Description(Analysis):
    def report(self, options: argparse.Namespace) -> None:
        self.model = Regression(self.reader.samples)
        print(f"{self.model.r('tach', 'engine')=:.3f}")
        print(
            f"engine = {self.model.b('tach', 'engine'):.2f} * tach + {self.model.a('tach', 'engine'):.0f}"
        )


class Model(Description):
    def report(self, options: argparse.Namespace) -> None:
        super().report(options)
        b = self.model.b("tach", "engine")
        a = self.model.a("tach", "engine")
        tach = range(1000, 3000, 200)
        print("| tach | engine")
        for t, e in zip(tach, map(lambda t: int(round(t * b + a, -2)), tach)):
            print(f"| {t} | {e}")


class FormatDetail(NamedTuple):
    row_generator: Type[RowGenerator]
    sample_builder: SampleBuilder
    additional: list[str]


class Configuration:
    FORMATS: dict[str, FormatDetail] = {
        "csv": FormatDetail(
            row_generator=csv.DictReader,
            sample_builder=lambda row: Sample(
                tach=float(row["Tach"]), engine=float(row["Engine"])
            ),
            additional=[],
        ),
        "ndjson": FormatDetail(
            row_generator=NDJSONRow,  # lambda source: (json.loads(line) for line in source),
            sample_builder=lambda row: Sample(tach=row["Tach"], engine=row["Engine"]),
            additional=[],
        ),
        "md": FormatDetail(
            row_generator=MDTableRow,
            sample_builder=lambda row: Sample(
                tach=float(row["Tach"]), engine=float(row["Engine"])
            ),
            additional=["table_index"],
        ),
        "toml": FormatDetail(
            row_generator=TOMLRow,
            sample_builder=lambda row: Sample(tach=row["Tach"], engine=row["Engine"]),
            additional=[],
        ),
    }
    COMMAND = Model


def get_options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        default=Path(os.environ.get("SOURCE") or "-"),
    )
    parser.add_argument("-f", "--format", choices=["csv", "ndjson", "toml", "md"])
    parser.add_argument(
        "-t",
        "--table",
        type=int,
        help="MD table number",
        dest="table_index",
        default=0,
    )
    options = parser.parse_args(argv)
    SUFFIX_MAP = {".csv": "csv", ".json": "ndjson", ".toml": "toml", ".md": "md"}
    if options.format is None:
        try:
            options.format = SUFFIX_MAP[options.source.suffix]
        except KeyError:
            parser.error(f"Unknown Suffix on {options.source}")
    return options


def main(argv: list[str] = sys.argv[1:]) -> None:
    command: Analysis = Configuration.COMMAND()
    command.run(Configuration, argv)


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
>>> f"{analysis.a('tach', 'engine')=:.3f}"
"analysis.a('tach', 'engine')=3.000"
>>> f"{analysis.b('tach', 'engine')=:.3f}"
"analysis.b('tach', 'engine')=0.500"
"""

test_config_1 = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text(
... 'Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68'
... )
>>> class ConfigLocal(Configuration):
...     pass
>>> options = get_options([str(data)])
>>> options
Namespace(source=PosixPath('temp.csv'), format='csv', table_index=0)
>>> rdr = build_reader(ConfigLocal, options)
>>> rdr.read(options.source)
>>> len(rdr.samples)
11
>>> analysis = Correlation(rdr.samples)
>>> f"{analysis.r('tach', 'engine')=:.3f}"
"analysis.r('tach', 'engine')=0.816"

"""

test_main = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text(
... 'Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68'
... )
>>> main(["temp.csv"])
self.model.r('tach', 'engine')=0.816
engine = 0.50 * tach + 3
| tach | engine
| 1000 | 500
| 1200 | 600
| 1400 | 700
| 1600 | 800
| 1800 | 900
| 2000 | 1000
| 2200 | 1100
| 2400 | 1200
| 2600 | 1300
| 2800 | 1400

"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}

if __name__ == "__main__":
    # import doctest
    #
    # doctest.testmod(verbose=True)
    main()
