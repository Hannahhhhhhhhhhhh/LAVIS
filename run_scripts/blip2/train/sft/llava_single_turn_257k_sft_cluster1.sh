GPUS_PER_NODE=2
export MASTER_PORT=8517
export CUDA_VISIBLE_DEVICES=5,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} train.py --cfg-path lavis/projects/blip2/train/llava_sft_st_257k_cluster.yaml