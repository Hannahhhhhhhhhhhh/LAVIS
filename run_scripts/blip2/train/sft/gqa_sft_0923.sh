GPUS_PER_NODE=2
export MASTER_PORT=8517
export CUDA_VISIBLE_DEVICES=3,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} train.py --cfg-path lavis/projects/blip2/train/sft/gqa_sft.yaml