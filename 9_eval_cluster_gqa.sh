#!/bin/bash
# remember to adjust output_dir!!!

# lizrun start -c "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/9_eval_cluster_gqa.sh" -n 1 -j test-blip2-lavis-cluster-gqa  -t all -i reg-ai.chehejia.com/ssai/lizr/cu118/py310/pytorch:2.0.1-multinode-nccl -p default

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
  pretrained: "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2-flant5-xxl/blip2_pretrained_flant5xxl.pth"
  
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

  preprocess_config: "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/configs/models/blip2/blip2_pretrain_flant5xxl.yaml"


datasets:
  gqa: # name of the dataset builder
    type: balanced_testdev
    vis_processor:
        eval:
          name: "blip_image_eval"
          image_size: 224
    text_processor:
        eval:
          name: "blip_question"
    build_info:
        images:
            storage: "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/images/"

run:
  task: gqa
  batch_size_train: 16
  batch_size_eval: 32
  num_workers: 4

  max_len: 500
  min_len: 1
  num_beams: 5
  inference_method: "generate"
  # prompt: "Question: {} Short answer:"
  prompt: ""

  seed: 42
  type: gqa
  output_dir: "/mnt/pfs-guan-ssai/nlu/wanghanzi/evaluation/BLIP2/GQA/llava_st_257k_sft_freaze_qf_textinf_cluster_kmeans_similar_ml500_0918/"
  result_output: "gqa_result.csv"
  split_save_path: "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/cluster/gqa_testdev_balanced_questions_infereance_0917/kmeans/split_samples/"

  amp: True
  resume_ckpt_path: null

  evaluate: False 
  train_splits: ["train"]

  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
EOT

torchrun --nnodes=1  --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP} \
    evaluate_cluster.py --cfg-path ${CONFIG_FILE}