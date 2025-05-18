#!/usr/bin/env bash

set -x

EXP_DIR=exps/vindr_calc_pos_lr2e-5_bsize1
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
