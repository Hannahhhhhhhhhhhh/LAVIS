GPUS_PER_NODE=2
export MASTER_PORT=8520
export CUDA_VISIBLE_DEVICES=5,1
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} train.py --cfg-path lavis/projects/blip2/train/prompt_moe/gqa_sft_prompt_moe.yaml