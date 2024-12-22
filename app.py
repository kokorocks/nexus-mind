from mergekit import merge_models

# Paths to the pretrained models
model_paths = ["path_to_model_1", "path_to_model_2"]

# Merge the models
merged_model = merge_models(model_paths)

# Save the merged model
merged_model.save_pretrained("path_to_merged_model")
