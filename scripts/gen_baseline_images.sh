#!/bin/sh

# GITHUB Actions version of gen_baseline_images.sh (using coverage as an intermediate between pytest and Coveralls)
coverage run --source=scedar -m pytest --mpl-generate-path=tests/baseline_images \
   --color=yes --ignore=tests/test_cluster/test_mirac_large_data.py tests

# TRAVIS version of gen_baseline_images.sh
#pytest --mpl-generate-path=tests/baseline_images \
#       --color=yes --cov-config .coveragerc \
#       --cov-branch --cov=scedar \
#       --ignore=tests/test_cluster/test_mirac_large_data.py \
#       tests
