# SOLID Python

Version 2.

This requires Python >= 3.9 and tox.

It's sometimes simpler to download Miniconda
and use this to install Python 3.9 or 3.10

See https://docs.conda.io/en/latest/miniconda.html

The ``conda`` tool is used to build
and activate environments.

Once that's done, Tox needs to be installed.
It's not generally part of the Anaconda 
environment, so a PIP install is required.

```sh
python -m pip install tox
```

After that the test suite can be run.

```sh
tox
```
