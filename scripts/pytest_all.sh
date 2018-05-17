#!/bin/sh
python setup.py test \
  --addopts "--mpl --mpl-baseline-path=tests/baseline_images \
             --color=yes --cov-config .coveragerc --cov-branch --cov=scedar"
