GPUS_PER_NODE=1
export MASTER_PORT=8517
export CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} train.py --cfg-path lavis/projects/blip2/train/llava_sft.yaml