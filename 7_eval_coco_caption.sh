#!/bin/bash

# lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/7_eval_coco_caption.sh" -n 1 -j test-blip2-lavis-coco  -t all -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p default

PROJECT_PATH=/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS
cd ${PROJECT_PATH}

export NCCL_IB_GID_INDEX=3

pip install -r requirements.txt

MASTER_IP=""
if [ "${RANK}" == "0" ];then
  while [[ "$MASTER_IP" == "" ]]
  do
    MASTER_IP=`ping ${MASTER_ADDR} -c 3 | sed '1{s/[^(]*(//;s/).*//;q}'`
    # MASTER_IP=127.0.0.1
    sleep 1
  done
else
  ## Convert DNS to IP for torch
  MASTER_IP=`getent hosts ${MASTER_ADDR} | awk '{print $1}'` # Ethernet
fi

# training cofiguration
CONFIG_FILE=/tmp/blip2_config_${RANK}.yaml
# WORLD_SIZE=`expr ${WORLD_SIZE} \* 8`
DIST_URL="env://${MASTER_IP}:${MASTER_PORT}"
# 配置生成


cat <<EOT > ${CONFIG_FILE}

model:
  arch: blip2_t5_instruct
  model_type: flant5xxl
  load_pretrained: True
  load_finetuned: False
  vit_model: eva_clip_g
  # intialize stage 2 pretraining from stage 1 pretrained model
  pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/instruct_blip_flant5xxl/instruct_blip_flanxxl_trimmed.pth"
  # pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/sft/llava_single_turn_257k_sft_unfreeze_qformer_0904/20230904145/checkpoint_2.pth"
  # pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/sft/llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0906/20230906210/checkpoint_2.pth"
  
  # vit encoder
  image_size: 224
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"
  freeze_vit: True

  # Q-Former
  num_query_token: 32
  freeze_qformer: True
  qformer_text_input: True

  # T5
  t5_model: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/google-flan-t5-xxl"
  prompt: ""

  max_txt_len: 256
  max_output_txt_len: 256

datasets:
  coco_caption: # name of the dataset builder
    vis_processor:
        eval:
          name: "blip_image_eval"
          # image_size: 364
          image_size: 224
    text_processor:
        eval:
          name: "blip_caption"

run:
  task: captioning
  # optimizer
  batch_size_train: 32
  batch_size_eval: 16
  num_workers: 4

  max_len: 30
  min_len: 8
  num_beams: 5

  seed: 42
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/blip2_flant5xxl_qf_instructblip_qt_0907"
  annotation_file: "/mnt/pfs-guan-ssai/nlu/dingyifeng/data/COCO/coco_karpathy_test.json" # coco

  evaluate: True
  test_splits: ["test"]

  # distribution-specific
  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True

EOT

torchrun --nnodes=1  --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP} \
    evaluate.py --cfg-path ${CONFIG_FILE}
