import os
import yaml
from easydict import EasyDict
import torch

from scenario_datasets import build_dataset

# Load config
cfg_file = "configs/test_config.yaml"
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
cfg = EasyDict(cfg)
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {cfg.device}")
print(f"\nTesting datasets: {cfg.datasets}\n")

# Test each dataset
for dataset_name in cfg.datasets:
    print(f"{'='*50}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*50}")
    
    try:
        dataset = build_dataset(
            dataset=dataset_name,
            root_path="./datasets",
            shots=cfg.num_shots
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Number of classes: {len(dataset.classnames)}")
        print(f"  - Class names (first 5): {dataset.classnames[:5]}")
        if hasattr(dataset, 'train_dataset'):
            print(f"  - Training samples: {len(dataset.train_dataset)}")
        if hasattr(dataset, 'test_dataset'):
            print(f"  - Test samples: {len(dataset.test_dataset)}")
        print()
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        print()

print(f"{'='*50}")
print("Dataset verification complete!")
print(f"{'='*50}")
