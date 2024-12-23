from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from mergekit.config import LoadConfig
from mergekit.merge import MergeOptions, ModelMerger

# Scenario 1: Merge similar sized models
merge_config_similar = {
    "models": {
        "pythia1": {
            "model": "EleutherAI/pythia-1.4b-deduped",
            "weight": 0.5
        },
        "pythia2": {
            "model": "EleutherAI/pythia-1.4b",
            "weight": 0.5
        }
    },
    "merge_method": "linear",
    "dtype": "float16",
    "output_path": "./merged_pythia"
}

# Scenario 2: Merge different sized models
merge_config_different = {
    "models": {
        "gpt2": {
            "model": "gpt2",
            "weight": 0.3
        },
        "gpt2medium": {
            "model": "gpt2-medium",
            "weight": 0.7
        }
    },
    "merge_method": "tied_mixture",
    "dtype": "float16",
    "output_path": "./merged_gpt2"
}

# Scenario 3: Merge specialized models
merge_config_specialized = {
    "models": {
        "code_model": {
            "model": "Salesforce/codegen-350M-mono",
            "weight": 0.6
        },
        "text_model": {
            "model": "facebook/opt-350m",
            "weight": 0.4
        }
    },
    "merge_method": "slerp",
    "dtype": "float16",
    "output_path": "./merged_specialized"
}

# Choose which configuration to use
current_config = merge_config_similar  # Change this to use different scenarios

# Save and execute merge
with open("merge_config.yaml", "w") as f:
    yaml.dump(current_config, f)

config = LoadConfig("merge_config.yaml")
merger = ModelMerger(config)
merger.merge()
