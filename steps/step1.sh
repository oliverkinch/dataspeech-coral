python main.py "oliverkinch/coral-tts-filtered" \
  --configuration "default" \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --rename_column \
  --repo_id "oliverkinch/coral-tts-filtered-tags" \
  --apply_squim_quality_estimation \
  --penn_batch_size 1024 \
  --batch_size 1
