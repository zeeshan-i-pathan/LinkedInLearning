"""S.O.L.I.D. Design.

Chapter 3, OCP. Part 4.

Library exposed to the configuration file
"""
import json

from dataclasses import dataclass
from typing import Any, Iterator, TextIO, Callable, NamedTuple


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


RowGenerator = Callable[..., Iterator[dict[str, Any]]]
SampleBuilder = Callable[[dict[str, Any]], Sample]


class ReaderConfig(NamedTuple):
    row_generator: RowGenerator
    sample_builder: SampleBuilder
    extensions: list[str]


def build_from_str(row: dict[str, Any]) -> Sample:
    return Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))


def ndjson_iter(input: TextIO) -> Iterator[dict[str, Any]]:
    yield from (json.loads(line) for line in input)


def build_from_float(row: dict[str, Any]) -> Sample:
    return Sample(tach=row["Tach"], engine=row["Engine"])
