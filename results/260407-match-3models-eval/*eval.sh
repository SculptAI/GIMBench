#!/bin/bash

set -x

# hf download Sculpt-AI/2604063-rebuttal-poisson-p0s1
# hf download Sculpt-AI/2604062-rebuttal-poisson-p1s0
# hf download Sculpt-AI/2604061-rebuttal-poisson-p1s1


python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2604063-rebuttal-poisson-p0s1 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2604062-rebuttal-poisson-p1s0 --output_type cfg --ref_model_device cpu --first_n 100
python -m gimbench.match.gim_regex --model_type vllm-offline --model_name Sculpt-AI/2604061-rebuttal-poisson-p1s1 --output_type cfg --ref_model_device cpu --first_n 100

shutdown -h +3
