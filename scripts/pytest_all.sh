#!/bin/sh

# GITHUB Actions version of pytest_all.sh (using coverage as an intermediate between pytest and Coveralls)
coverage run --source=scedar -m pytest --mpl \
     --mpl-baseline-path=tests/baseline_images --color=yes tests

# TRAVIS version of pytest_all.sh
#pytest --mpl --mpl-baseline-path=tests/baseline_images \
#       --color=yes --cov-config .coveragerc --cov-branch --cov=scedar \
#       tests
