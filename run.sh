#!/bin/sh

set -e  # If any command failed, exit directly

export PYTHONPATH=$PWD:${PYTHONPATH}

OUTPUT_DIR=$PWD/output
mkdir -p $OUTPUT_DIR

## 训练
DATA_DIR=$PWD/wide_deep_demo/dataset/adult

python $PWD/wide_deep_demo/wide_deep_demo.py --data_dir ${DATA_DIR} --model_dir ${OUTPUT_DIR}/ckpt \
                         --train_epochs 40 --export_dir ${OUTPUT_DIR}/saved_model

##　裁剪模型
## step1: extract_vars_from_saved_model
## 从已经保存的模型目录，导出ckpt文件到output_dir
## step2: extract_gftrl_embeddings
## 从feature_spec文件读取feature spec，读取embedding的非0值shape，写入spec到output_dir
mkdir -p ${OUTPUT_DIR}/transformed_ckpt/
python -m opal.tensorflow.optimizer.gftrl_embedding_transform \
    --input-savedmodel-dir ${OUTPUT_DIR}/saved_model/*/  \
    --output-checkpoint-dir ${OUTPUT_DIR}/transformed_ckpt/ \
    --feature-json-spec-file $PWD/wide_deep_demo/embedding_transform_spec.json
ls ${OUTPUT_DIR}/transformed_ckpt/

## 重新导出模型
## export_savedmodel 导出SavedModel
mkdir -p ${OUTPUT_DIR}/transformed_model
python $PWD/wide_deep_demo/wide_deep_demo.py --data_dir ${DATA_DIR} --model_dir ${OUTPUT_DIR}/transformed_ckpt \
                         --train_epochs 40 --export_dir ${OUTPUT_DIR}/transformed_model --run_mode  export_model
ls ${OUTPUT_DIR}/transformed_model/
