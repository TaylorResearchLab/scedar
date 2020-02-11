# Update Package

Things need to be changed/checked:

- Version number:
  - `setup.py`
  - `CHANGELOG.md`
  - `docs/conf.py`
- Text:
  - `CHANGELOG.md`
  - `README.md`
  - `pypiREADME.md`
  - build docs
  - documentation in source files
  - variable names in source files

Release using [twine](https://packaging.python.org/tutorials/packaging-projects/).

```bash
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
python3 -m pip install --user --upgrade twine
python3 -m twine upload dist/*
```
