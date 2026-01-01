# 格式转换
# python convert_to_qwen_format.py \
#   --input output_dataset.jsonl \
#   --dataset_root /opt/data/private/gaoj/GaoJing/curriculum/Fundamentals_and_Applications_of_Large_Models/Model_Finetune/dataset \
#   --output_train train.jsonl

# 主体微调框架
CUDA_VISIBLE_DEVICES=3 \
MAX_PIXELS=1605632 swift sft \
--model ./cache/Qwen/Qwen3-VL-2B-Instruct \
--dataset ./train3.jsonl \
--train_type lora \
--lorap_lr_ratio 10 \
--freeze_vit false \
--freeze_aligner false \
--freeze_llm false \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--split_dataset_ratio 0.1 \
--output_dir ./output \
--num_train_epochs 6 \
--save_steps 20 \
--eval_steps 20 \
--save_total_limit 2 \
--logging_steps 10 \
--seed 42 \
--learning_rate 1e-4 \
--init_weights true \
--lora_rank 8 \
--lora_alpha 32 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-08 \
--weight_decay 0.1 \
--gradient_accumulation_steps 16 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--warmup_steps 0 \
--gradient_checkpointing True

#可视化分析
python render_pred_boxes.py \
  --dataset_jsonl ./output/v7-20251226-145826/val_dataset.jsonl \
  --out_dir vis_eval \
  --skip_missing 
