#!/bin/sh
pytest --mpl --mpl-baseline-path=tests/baseline_images \
       --color=yes --cov-config .coveragerc --cov-branch --cov=scedar \
       tests
