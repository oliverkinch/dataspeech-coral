python ./scripts/run_prompt_creation.py \
  --is_new_speaker_prompt \
  --dataset_name "oliverkinch/coral-tts-filtered-tags-small" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 8 \
  --attn_implementation "sdpa" \
  --output_dir "./tmp" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "oliverkinch/coral-tts-filtered-tagged-small" \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8
