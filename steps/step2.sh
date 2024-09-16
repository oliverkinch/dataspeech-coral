python ./scripts/metadata_to_text.py \
    "oliverkinch/coral-tts-filtered-tags-small" \
    --repo_id "oliverkinch/coral-tts-filtered-tags-small" \
    --configuration "default" \
    --cpu_num_workers "8" \
    --path_to_bin_edges "./examples/tags_to_annotations/v02_bin_edges.json" \
    --path_to_text_bins "./examples/tags_to_annotations/v02_text_bins.json" \
    --apply_squim_quality_estimation