#!/bin/sh
pytest --mpl-generate-path=tests/baseline_images \
       --color=yes --cov-config .coveragerc \
       --cov-branch --cov=scedar \
       --ignore=tests/test_cluster/test_mirac_large_data.py \
       tests
