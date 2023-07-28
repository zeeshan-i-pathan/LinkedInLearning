"""S.O.L.I.D. Design.

Chapter 3, OCP. Part 4.

Configuration file.
"""

import csv
import json

from ex03_lib import *


class Configuration:
    CSV = ReaderConfig(
        row_generator=csv.DictReader,
        sample_builder=lambda row: Sample(
            tach=float(row["Tach"]), engine=float(row["Engine"])
        ),
        extensions=[".csv"],
    )
    NDJSON = ReaderConfig(
        row_generator=lambda input: (json.loads(line) for line in input),
        sample_builder=lambda row: Sample(tach=row["Tach"], engine=row["Engine"]),
        extensions=[".json", ".ndjson"],
    )
    # TOML = ReaderConfig(...)
    # MD = ReaderConfig(...)
