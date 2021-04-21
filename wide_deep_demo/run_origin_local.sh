#!/bin/sh

export PYTHONPATH=$PWD:${PYTHONPATH}

DATA_DIR=$PWD/dataset/adult
OUTPUT_DIR=$PWD/output
EXPORT_DIR=$PWD/output

python wide_deep_demo_origin.py --data_dir ${DATA_DIR} --model_dir ${OUTPUT_DIR} \
                         --train_epochs 40 --export_dir ${EXPORT_DIR}
