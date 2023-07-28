"""S.O.L.I.D. Design.

Chapter 1, ISP. Part 4.
"""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

from dataclasses import InitVar, dataclass, field
from typing import Iterable, Iterator, Optional, Union, overload


@dataclass(frozen=True)
class Sample:
    tach: float
    engine: float


class SampleBuilder:
    @staticmethod
    def sample(row: dict[str, str]) -> Sample:
        return Sample(tach=float(row["Tach"]), engine=float(row["Engine"]))


class SampleList:
    def __init__(self, samples: Optional[Iterable[Sample]] = None) -> None:
        self.samples = list(samples) if samples else []

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @overload
    def __getitem__(self, index: int) -> Sample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[Sample]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Sample, list[Sample]]:
        return self.samples[index]


class Reader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.samples: SampleList = SampleList()

    def read(self) -> None:
        with self.source.open() as input:
            reader = csv.DictReader(input)
            self.samples = SampleList(SampleBuilder.sample(row) for row in reader)


class Correlation:
    def __init__(self, reader: Reader) -> None:
        self._samples = reader.samples

    def r(self) -> float:
        self.mean_x = mean(s.tach for s in self._samples)
        self.mean_y = mean(s.engine for s in self._samples)
        sd_x = stdev(s.tach for s in self._samples)
        sd_y = stdev(s.engine for s in self._samples)
        n = len(self._samples)
        r = (
            sum(s.tach * s.engine for s in self._samples)
            - n * self.mean_x * self.mean_y
        ) / ((n - 1) * sd_x * sd_y)
        return r


test_sample = """
>>> row = {"Tach": "10", "Engine": "8.04"}
>>> s_0 = SampleBuilder.sample(row)
>>> s_0
Sample(tach=10.0, engine=8.04)
"""

test_reader = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> reader = Reader(data)
>>> reader.read()
>>> len(reader.samples)
11
>>> reader.samples[0]
Sample(tach=10.0, engine=8.04)
>>> reader.samples[-1]
Sample(tach=5.0, engine=5.68)
"""

test_correlation = """
>>> from pathlib import Path
>>> data = Path("temp.csv")
>>> _ = data.write_text('Tach,Engine\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> reader = Reader(data)
>>> reader.read()
>>> analysis = Correlation(reader)
>>> f"{analysis.r()=:.3f}"
'analysis.r()=0.816'
"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}

if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
