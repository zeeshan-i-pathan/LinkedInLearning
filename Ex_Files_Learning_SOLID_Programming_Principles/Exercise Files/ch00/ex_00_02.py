"""S.O.L.I.D. Python Class Design.

Chapter 0, Introduction. Part 2.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev


class Correlation:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.data: list[dict[str, str]] = []

    def read(self) -> None:
        with self.source.open() as input:
            reader = csv.DictReader(input)
            self.data = list(reader)

    def x_values(self, data: list[dict[str, str]]) -> list[str]:
        return [row["x"] for row in data]

    def y_values(self, data: list[dict[str, str]]) -> list[str]:
        return [row["y"] for row in data]

    def r(self, x_values: list[str], y_values: list[str]) -> float:
        self.mean_x = mean(float(x) for x in x_values)
        self.mean_y = mean(float(y) for y in y_values)
        sd_x = stdev(float(x) for x in x_values)
        sd_y = stdev(float(y) for y in y_values)
        n = len(x_values)
        r = (
            sum(float(x) * float(y) for x, y in zip(x_values, y_values))
            - n * self.mean_x * self.mean_y
        ) / ((n - 1) * sd_x * sd_y)
        return r


def regression_1(corr: Correlation) -> tuple[float, float]:
    x_values = corr.x_values(corr.data)
    y_values = corr.y_values(corr.data)

    mean_x = mean(float(x) for x in x_values)
    mean_y = mean(float(y) for y in y_values)
    sd_x = stdev(float(x) for x in x_values)
    sd_y = stdev(float(y) for y in y_values)
    n = len(x_values)

    b = (
        sum(float(x) * float(y) for x, y in zip(x_values, y_values))
        - n * mean_x * mean_y
    ) / ((n - 1) * sd_x * sd_x)
    a = mean_y - b * mean_x
    return a, b


def regression(corr: Correlation) -> tuple[float, float]:
    x_values = corr.x_values(corr.data)
    y_values = corr.y_values(corr.data)

    mean_x = mean(float(x) for x in x_values)
    mean_y = mean(float(y) for y in y_values)
    sd_x = stdev(float(x) for x in x_values)
    sd_y = stdev(float(y) for y in y_values)

    b = corr.r(x_values, y_values) * sd_y / sd_x
    a = mean_y - b * mean_x
    return a, b


def main(source: Path) -> None:
    analysis = Correlation(source)
    analysis.read()
    x = analysis.x_values(analysis.data)
    y = analysis.y_values(analysis.data)
    print(f"Correlation = {analysis.r(x, y):.3f}")


test_reader = """
>>> from pathlib import Path
>>> t = Path("temp.csv")
>>> _ = t.write_text('x,y\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> reader = Correlation(t)
>>> reader.read()
>>> reader.data
[{'x': '10.0', 'y': '8.04'}, {'x': '8.0', 'y': '6.95'}, {'x': '13.0', 'y': '7.58'}, {'x': '9.0', 'y': '8.81'}, {'x': '11.0', 'y': '8.33'}, {'x': '14.0', 'y': '9.96'}, {'x': '6.0', 'y': '7.24'}, {'x': '4.0', 'y': '4.26'}, {'x': '12.0', 'y': '10.84'}, {'x': '7.0', 'y': '4.82'}, {'x': '5.0', 'y': '5.68'}]

"""

test_correlation = """
>>> from pathlib import Path
>>> t = Path("temp.csv")
>>> _ = t.write_text('x,y\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> analysis = Correlation(t)
>>> analysis.read()
>>> x = analysis.x_values(analysis.data)
>>> y = analysis.y_values(analysis.data)
>>> f"{analysis.r(x, y)=:.3f}"
'analysis.r(x, y)=0.816'
"""

test_main = """
>>> from pathlib import Path
>>> t = Path("temp.csv")
>>> _ = t.write_text('x,y\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> main(t)
Correlation = 0.816
"""


test_regression = """
>>> from pathlib import Path
>>> t = Path("temp.csv")
>>> _ = t.write_text('x,y\\n10.0,8.04\\n8.0,6.95\\n13.0,7.58\\n9.0,8.81\\n11.0,8.33\\n14.0,9.96\\n6.0,7.24\\n4.0,4.26\\n12.0,10.84\\n7.0,4.82\\n5.0,5.68')
>>> analysis = Correlation(t)
>>> analysis.read()
>>> x = analysis.x_values(analysis.data)
>>> y = analysis.y_values(analysis.data)
>>> f"{analysis.r(x, y)=:.3f}"
'analysis.r(x, y)=0.816'
>>> a, b = regression(analysis)
>>> f"y = {b:.3f}x + {a:.3f}"
'y = 0.500x + 3.000'
"""

__test__ = {name: value for name, value in locals().items() if name.startswith("test")}


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
