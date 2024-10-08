#!/bin/bash
set -ex
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

# encoder1

model_transform.py \
    --model_name audio_encoder_1 \
    --model_def ../audio_encoder_1.onnx \
    --test_input ../input_encoder_1.npz \
    --test_result audio_encoder_1_top_output.npz \
    --input_shapes [1,262144] \
    --mlir audio_encoder_1.mlir \
    --dynamic \
    --debug


model_deploy.py \
    --mlir audio_encoder_1.mlir \
    --quantize F32 \
    --test_input ../input_encoder_1.npz \
    --test_reference audio_encoder_1_top_output.npz \
    --chip bm1684x \
    --model audio_encoder_1.bmodel \
    --disable_layer_group \
    --compare_all \
    --dynamic

# encoder2

model_transform.py \
    --model_name audio_encoder_2 \
    --model_def ../audio_encoder_2.onnx \
    --test_input ../input_encoder_2.npz \
    --test_result audio_encoder_2_top_output.npz \
    --input_shapes [[1,490,512],[1,8]] \
    --mlir audio_encoder_2.mlir \
    --dynamic \
    --debug


model_deploy.py \
    --mlir audio_encoder_2.mlir \
    --quantize F32 \
    --test_input ../input_encoder_2.npz \
    --test_reference audio_encoder_2_top_output.npz \
    --chip bm1684x \
    --model audio_encoder_2.bmodel \
    --disable_layer_group \
    --compare_all \
    --dynamic


# ppe
model_transform.py \
    --model_name ppe \
    --model_def ../../onnx/ppe.onnx \
    --input_shapes [[1,490,64]] \
    --mlir ppe.mlir \
    --dynamic

model_deploy.py \
    --mlir ppe.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model ppe.bmodel \
    --compare_all \
    --dynamic

# decoder
model_transform.py \
    --model_name decoder \
    --model_def ../../onnx/decoder.onnx \
    --input_shapes [[1,490,64],[1,490,64],[490,490]] \
    --mlir decoder.mlir \
    --dynamic


model_deploy.py \
    --mlir decoder.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model decoder.bmodel \
    --compare_all \
    --dynamic
    
popd