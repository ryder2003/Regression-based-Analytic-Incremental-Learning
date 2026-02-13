# Primal RAIL Code Flow Guide

This document provides a detailed walkthrough of the Primal RAIL (Regression-based Analytic Incremental Learning) implementation, explaining the code flow, algorithm steps, and key components.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [File Structure](#file-structure)
3. [Execution Flow](#execution-flow)
4. [Core Components](#core-components)
5. [Algorithm Deep Dive](#algorithm-deep-dive)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Key Functions Explained](#key-functions-explained)

---

## High-Level Overview

### What is Primal RAIL?

Primal RAIL is a continual learning method that:
- Learns multiple tasks sequentially without forgetting previous knowledge
- Uses **analytic learning** (closed-form solutions) instead of gradient descent
- Employs **recursive ridge regression** for efficient weight updates
- Fuses zero-shot CLIP predictions with learned task-specific adapters

### Algorithm in 3 Steps

```
1. Extract features using frozen CLIP encoder
2. Expand features through ReLU non-linearity
3. Update adapter weights using recursive ridge regression
```

---

## File Structure

### Main Files

```
primal_RAIL.py              # Main training script (entry point)
dual_RAIL.py                # Alternative dual-form implementation
utils.py                    # Helper functions (cache, evaluation)
configs/
  └── test_config.yaml      # Configuration file
scenario_datasets/
  ├── __init__.py           # Dataset registry
  ├── utils.py              # Data loading utilities
  ├── dtd.py                # DTD dataset wrapper
  ├── oxford_flowers.py     # Oxford Flowers wrapper
  ├── oxford_pets.py        # Oxford Pets wrapper
  └── mnist.py              # MNIST wrapper
clip/
  ├── __init__.py
  ├── clip.py               # CLIP model loading
  └── model.py              # CLIP architecture
```

---

## Execution Flow

### Phase 1: Initialization (Lines 137-154)

```python
if __name__ == '__main__':
    # 1. Load configuration
    cfg = setup_config()  # Read test_config.yaml
    
    # 2. Set random seeds
    seed_everything(cfg.seed)  # Reproducibility
    
    # 3. Initialize CLIP model
    clip_model, preprocess = load_clip()  # Load ViT-B/16
    
    # 4. Build datasets
    datasets = build_continual_datasets(cfg)  # DTD, Flowers, Pets
```

**What Happens:**
- Configuration loaded from `configs/test_config.yaml`
- Random seeds set for NumPy, PyTorch, CUDA (seed=1)
- CLIP ViT-B/16 model loaded (frozen weights)
- Dataset objects created for all 3 tasks

---

### Phase 2: Feature Extraction (Lines 155-161)

```python
# Extract CLIP features for all datasets
train_features, train_labels, test_features, test_labels = [], [], [], []
for dataset in datasets:
    train_feat, train_lab = extract_clip_features(dataset.train_loader, clip_model)
    test_feat, test_lab = extract_clip_features(dataset.test_loader, clip_model)
    train_features.append(train_feat)  # Shape: [N, 512]
    test_features.append(test_feat)
```

**What Happens:**
- Each image passed through CLIP encoder: `Image → [512-dim vector]`
- Features cached to avoid re-computation
- Zero-shot predictions computed: `features @ text_embeddings.T`
- All features stored in memory (efficient for few-shot scenarios)

**Key Function:** `extract_clip_features()`
```python
def extract_clip_features(loader, clip_model):
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            feats = clip_model.encode_image(images)  # [B, 512]
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            features.append(feats)
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)
```

---

### Phase 3: Continual Learning Loop (Lines 162-299)

#### Task Loop Structure

```python
for task_id in range(len(datasets)):
    print(f"Task {task_id + 1}: {dataset_names[task_id]}")
    
    # Step 1: Train on current task
    train_adapter(task_id)
    
    # Step 2: Evaluate on all tasks seen so far
    for eval_task_id in range(task_id + 1):
        evaluate_task(eval_task_id)
    
    # Step 3: Store results
    save_results(task_id)
```

---

### Phase 4: Training (Analytic Adapter Update)

#### Task 1 Initialization (Lines 166-186)

```python
if task_id == 0:  # First task
    # 1. Expand features through ReLU
    train_features_expanded = F.relu(train_features[0] @ Rand_Proj)
    # Shape: [N, 512] @ [512, 15000] → [N, 15000]
    
    # 2. Compute auto-correlation matrix
    auto_cor = train_features_expanded.T @ train_features_expanded
    # Shape: [15000, 15000]
    
    # 3. Compute inverse (ridge regression)
    R = (auto_cor + λI)^(-1)
    # Shape: [15000, 15000]
    
    # 4. Compute adapter weights
    W = R @ (train_features_expanded.T @ train_labels_onehot)
    # Shape: [15000, num_classes_task_0]
```

**Mathematical Explanation:**
- **Goal:** Solve `W = argmin ||XW - Y||^2 + λ||W||^2` (ridge regression)
- **Closed-form solution:** `W = (X^T X + λI)^(-1) X^T Y`
- **R-matrix:** Stores `(X^T X + λI)^(-1)` for efficient updates

---

#### Task 2+ Update (Lines 188-225)

```python
else:  # Subsequent tasks (t > 0)
    # 1. Expand new task features
    train_features_expanded = F.relu(train_features[task_id] @ Rand_Proj)
    
    # 2. Recursive update of R-matrix (Sherman-Morrison-Woodbury)
    R_new = R - R @ X_new.T @ (I + X_new @ R @ X_new.T)^(-1) @ X_new @ R
    # Avoids recomputing full inverse!
    
    # 3. Expand adapter to include new classes
    W_expanded = pad_weights(W, new_classes)
    # Shape: [15000, total_classes_so_far]
    
    # 4. Update weights
    W_new = W_expanded + R_new @ (X_new.T @ Y_new_residual)
```

**Key Innovation:**
- **Recursive formula** avoids recomputing matrix inverse
- Complexity: O(d²n) instead of O(d³)
- **Memory efficient:** Only store R-matrix (d × d)

---

### Phase 5: Evaluation (Lines 227-289)

```python
for eval_task_id in range(task_id + 1):
    # 1. Expand test features
    test_features_expanded = F.relu(test_features[eval_task_id] @ Rand_Proj)
    
    # 2. Get adapter predictions
    adapter_logits = test_features_expanded @ W
    # Shape: [N_test, total_classes]
    
    # 3. Get zero-shot predictions
    zeroshot_logits = test_features[eval_task_id] @ text_embeddings.T
    # Shape: [N_test, total_classes]
    
    # 4. Fusion
    final_logits = (1 - α) * zeroshot_logits + α * adapter_logits
    # α = 0.8 (fusion weight)
    
    # 5. Extract task-specific predictions
    task_logits = extract_task_columns(final_logits, eval_task_id)
    
    # 6. Compute accuracy
    predictions = torch.argmax(task_logits, dim=1)
    accuracy = (predictions == labels).float().mean()
```

---

## Core Components

### 1. Configuration (`setup_config()`)

```yaml
# configs/test_config.yaml
root_path: "datasets"
datasets: ["dtd", "oxford_flowers", "oxford_pets"]
shots: 16
batch_size: 64
batch_size_eval: 256
hidden_dim: 15000
regularization: 0.1
fusion_weight: 0.8
seed: 1
```

**Loaded into EasyDict:**
```python
cfg.root_path          # Dataset root directory
cfg.datasets           # List of dataset names
cfg.shots              # K-shot per class
cfg.hidden_dim         # Expanded feature dimension
cfg.regularization     # Ridge penalty (λ)
cfg.fusion_weight      # Alpha for fusion
```

---

### 2. CLIP Model (`load_clip_to_cpu()`)

```python
def load_clip_to_cpu():
    url = "ViT-B/16"  # Model variant
    model_path = download_clip(url)  # Download if needed
    model = build_model(model_path)  # Build architecture
    return model
```

**Architecture:**
```
Input Image (224x224x3)
    ↓
Patch Embedding (16x16 patches)
    ↓
Vision Transformer (12 layers)
    ↓
Layer Normalization
    ↓
Output: 512-dim feature vector
```

---

### 3. Dataset Builder (`build_dataset()`)

```python
def build_dataset(dataset_name, root_path, shots):
    # scenario_datasets/__init__.py
    dataset_list = {
        "dtd": DTD,
        "oxford_flowers": OxfordFlowers,
        "oxford_pets": OxfordPets,
        # ...
    }
    dataset_class = dataset_list[dataset_name]
    return dataset_class(root_path, shots)
```

**Dataset Wrapper Structure:**
```python
class DatasetBase:
    def __init__(self, root_path, shots):
        self.train_x = []  # List of (image_path, label)
        self.train_u = []  # Unlabeled data (unused)
        self.val = []      # Validation split
        self.test = []     # Test split
        
    def get_lab2cname(self):
        return {0: "class_0", 1: "class_1", ...}
        
    def generate_fewshot_dataset(self, shots):
        # Sample K shots per class
        pass
```

---

### 4. Continual CLIP Adapter (`continual_clip_adaptor`)

```python
class continual_clip_adaptor(nn.Module):
    def __init__(self, cfg, clip_model):
        self.cfg = cfg
        self.clip_model = clip_model  # Frozen
        self.Rand_Proj = self.initialize_random_projection()
        self.R = None  # R-matrix for recursive updates
        self.W = None  # Adapter weights
        self.num_classes_seen = 0
        
    def initialize_random_projection(self):
        # Xavier initialization
        d_in = 512  # CLIP feature dim
        d_hidden = cfg.hidden_dim  # 15000
        scale = (2 / (d_in + d_hidden)) ** 0.5
        return torch.randn(d_in, d_hidden) * scale
```

---

## Algorithm Deep Dive

### Recursive Ridge Regression

**Problem:** As we learn task t, we want to update weights without forgetting tasks 1...t-1.

**Standard Ridge Regression:**
```
W = (X^T X + λI)^(-1) X^T Y
```

**For Task 0:**
```python
X_0 = train_features_expanded  # [N_0, d]
Y_0 = train_labels_onehot       # [N_0, C_0]
R_0 = (X_0^T X_0 + λI)^(-1)     # [d, d]
W_0 = R_0 @ X_0^T @ Y_0         # [d, C_0]
```

**For Task t (t > 0):**
```python
# Concatenate all data seen so far
X_cumulative = [X_0; X_1; ...; X_t]  # [N_total, d]
Y_cumulative = [Y_0; Y_1; ...; Y_t]  # [N_total, C_total]

# Naive approach: Recompute inverse (expensive!)
R_t = (X_cumulative^T X_cumulative + λI)^(-1)  # O(d³)

# Efficient approach: Recursive update
R_t = R_{t-1} - R_{t-1} @ X_t^T @ (I + X_t @ R_{t-1} @ X_t^T)^(-1) @ X_t @ R_{t-1}
# Only O(d² n_t) where n_t = samples in task t
```

**Implementation:**
```python
# Line 197-206
A = Rt @ Xt.T  # [d, n_t]
B = Xt @ A     # [n_t, n_t]
C = B + I_nt   # [n_t, n_t]
C_inv = torch.inverse(C)
D = A @ C_inv  # [d, n_t]
E = D @ Xt     # [d, d]
Rt_new = Rt - E @ Rt
```

**Why This Works:**
- Based on **Woodbury Matrix Identity**:
  ```
  (A + UCV)^(-1) = A^(-1) - A^(-1)U(C^(-1) + VA^(-1)U)^(-1)VA^(-1)
  ```
- Allows updating inverse incrementally
- Crucial for continual learning efficiency

---

### Weight Expansion

When learning a new task with new classes, we need to expand the weight matrix:

```python
# Before Task t:
W_{t-1} = [W_0 | W_1 | ... | W_{t-1}]  # [d, C_0 + C_1 + ... + C_{t-1}]

# After Task t:
W_t = [W_0 | W_1 | ... | W_{t-1} | W_t]  # [d, C_0 + ... + C_t]
```

**Implementation:**
```python
# Line 212-220
new_classes = dataset.num_classes
old_classes = W.shape[1]
total_classes = old_classes + new_classes

# Expand weight matrix
W_expanded = torch.zeros(hidden_dim, total_classes)
W_expanded[:, :old_classes] = W  # Copy old weights
# New class weights initialized to 0
```

---

### Fusion Mechanism

Combines two complementary predictions:

1. **Zero-shot (CLIP):** Generalizes well but may lack task-specific discrimination
2. **Adapter:** Task-specific but may overfit to few-shot data

```python
# Line 263-268
zeroshot_logits = test_features @ text_embeddings.T
adapter_logits = test_features_expanded @ W
final_logits = (1 - α) * zeroshot_logits + α * adapter_logits
```

**Fusion Weight (α = 0.8):**
- 80% adapter contribution
- 20% zero-shot contribution
- Balances generalization and specialization

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     INITIALIZATION PHASE                     │
└─────────────────────────────────────────────────────────────┘
    ↓
Load Config (test_config.yaml)
    ↓
Load CLIP Model (ViT-B/16) - Frozen
    ↓
Build Datasets (DTD, Flowers, Pets)
    ↓
Extract All Features (Cached)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   CONTINUAL LEARNING LOOP                    │
└─────────────────────────────────────────────────────────────┘

FOR each task t in [0, 1, 2]:
    
    ┌─────────────────────────────────────────────────────────┐
    │                    TRAINING PHASE                        │
    └─────────────────────────────────────────────────────────┘
    
    Load Train Features (Task t)
        ↓
    Expand Features: X_t @ Rand_Proj → X_t_expanded
        ↓
    Apply ReLU: X_t_expanded = relu(X_t_expanded)
        ↓
    IF t == 0:
        Compute R_0 = (X_0^T X_0 + λI)^(-1)
        Compute W_0 = R_0 @ X_0^T @ Y_0
    ELSE:
        Update R_t using recursive formula
        Expand W to include new classes
        Update W_t = W_{t-1} + ΔW
        ↓
    
    ┌─────────────────────────────────────────────────────────┐
    │                   EVALUATION PHASE                       │
    └─────────────────────────────────────────────────────────┘
    
    FOR each previous task j in [0, 1, ..., t]:
        
        Load Test Features (Task j)
            ↓
        Expand Test Features: relu(X_test_j @ Rand_Proj)
            ↓
        Compute Adapter Logits: X_test_j_expanded @ W_t
            ↓
        Compute Zero-shot Logits: X_test_j @ Text_Embeddings
            ↓
        Fusion: (1-α) * ZS + α * Adapter
            ↓
        Extract Task-specific Columns
            ↓
        Compute Accuracy
            ↓
        Store Results
    
    END FOR
    
END FOR

┌─────────────────────────────────────────────────────────────┐
│                      RESULTS DISPLAY                         │
└─────────────────────────────────────────────────────────────┘

Print Accuracy Matrix
Calculate Average Metrics
```

---

## Key Functions Explained

### 1. `extract_features_clip()` (utils.py, Line 15-45)

**Purpose:** Extract 512-dim CLIP features from images

**Flow:**
```python
def extract_features_clip(loader, clip_model, cache_keys):
    # 1. Check cache first
    if cache_exists(cache_keys):
        return load_from_cache(cache_keys)
    
    # 2. Extract features
    features, labels = [], []
    for images, targets in tqdm(loader):
        with torch.no_grad():
            images = images.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu())
        labels.append(targets)
    
    # 3. Save to cache
    features = torch.cat(features)
    labels = torch.cat(labels)
    save_to_cache(features, labels, cache_keys)
    
    return features, labels
```

**Cache Keys:**
```python
cache_keys = [
    'few_shot',  # Data split type
    cfg.dataset,  # Dataset name
    cfg.backbone,  # Model architecture
    cfg.shots      # K-shot value
]
```

---

### 2. `zeroshot_classifier()` (utils.py, Line 90-110)

**Purpose:** Create text embeddings for class names

**Flow:**
```python
def zeroshot_classifier(classnames, templates, clip_model):
    # 1. Generate text prompts
    texts = []
    for classname in classnames:
        for template in templates:
            texts.append(template.format(classname))
    # Example: "a photo of a banded texture"
    
    # 2. Tokenize
    text_tokens = clip.tokenize(texts).cuda()
    
    # 3. Encode with CLIP text encoder
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 4. Average over templates
    text_features = text_features.reshape(len(classnames), len(templates), -1)
    text_features = text_features.mean(dim=1)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features  # [num_classes, 512]
```

**Templates:**
```python
# For ImageNet-style datasets
templates = [
    'a photo of a {}.',
    'a bad photo of a {}.',
    'a photo of many {}.',
    # ... more variations
]
```

---

### 3. `build_data_loader()` (scenario_datasets/utils.py, Line 485-510)

**Purpose:** Create PyTorch DataLoader with preprocessing

**Flow:**
```python
def build_data_loader(data_source, batch_size, is_train, shuffle):
    # 1. Define transforms
    if is_train:
        transform = Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    else:
        transform = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD)
        ])
    
    # 2. Create dataset
    dataset = DatasetWrapper(data_source, transform)
    
    # 3. Create loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    return loader
```

---

### 4. `cls_acc()` (utils.py, Line 140-155)

**Purpose:** Calculate top-1 accuracy

**Flow:**
```python
def cls_acc(output, target):
    # output: [N, C] logits
    # target: [N] labels
    
    predictions = torch.argmax(output, dim=1)
    correct = (predictions == target).float()
    accuracy = correct.mean() * 100.0
    
    return accuracy
```

---

## Advanced Topics

### Random Projection Matrix

**Initialization (Line 60-65):**
```python
def initialize_random_projection(input_dim, hidden_dim):
    # Xavier/Glorot initialization
    std = (2.0 / (input_dim + hidden_dim)) ** 0.5
    W_random = torch.randn(input_dim, hidden_dim) * std
    return W_random
```

**Why Random Projection?**
- **Feature expansion:** 512 → 15,000 dimensions
- **Non-linear transformation:** followed by ReLU
- **Fixed (not learned):** Reduces parameters
- **Theoretical basis:** Random Kitchen Sinks, Neural Tangent Kernel

---

### Memory Management

**Efficient Caching:**
```python
# Cache directory structure
caches/
  ├── dtd_ViT-B16_16shots_train.pt
  ├── dtd_ViT-B16_16shots_test.pt
  ├── oxford_flowers_ViT-B16_16shots_train.pt
  └── ...
```

**Loading Strategy:**
```python
# First run: Extract and cache (slow)
features = extract_features_clip(loader, clip_model)  # ~2 min per dataset

# Subsequent runs: Load from cache (fast)
features = torch.load(cache_path)  # <1 second
```

---

### Regularization Parameter (λ)

**Effect on Learning:**
- **λ = 0:** No regularization (may overfit)
- **λ = 0.1:** Moderate regularization (used in config)
- **λ = 1.0:** Strong regularization (may underfit)

**Mathematical Role:**
```python
# Ridge penalty prevents singular matrices
R = (X^T X + λI)^(-1)

# Without λ, X^T X may not be invertible
# With λ, always invertible (adds to diagonal)
```

---

## Debugging Tips

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce `batch_size` in config (64 → 32)
   - Reduce `hidden_dim` (15000 → 10000)

2. **Slow Training:**
   - Check if features are cached
   - Verify GPU is being used: `torch.cuda.is_available()`

3. **Poor Accuracy:**
   - Check `fusion_weight` (try 0.5-0.9)
   - Verify dataset splits loaded correctly
   - Check `regularization` (try 0.01-1.0)

4. **Import Errors:**
   - Ensure virtual environment activated
   - Run: `pip install -r requirements.txt`

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Time (RTX 3050) |
|-----------|-----------|-----------------|
| CLIP Feature Extraction | O(N) | ~1-2 min/dataset |
| Feature Expansion | O(Nd²) | ~1 second |
| R-matrix Update (Task 0) | O(d³) | ~3 seconds |
| R-matrix Update (Task t>0) | O(d²n) | ~1 second |
| Weight Update | O(d²C) | <1 second |
| Evaluation | O(NdC) | ~1-2 min/dataset |

Where:
- N = number of samples
- d = hidden dimension (15,000)
- C = number of classes

### Memory Usage

```
CLIP Model:         ~350 MB
Random Projection:  ~30 MB (512 × 15000 × 4 bytes)
R-matrix:           ~900 MB (15000² × 4 bytes)
Adapter Weights:    ~11 MB (15000 × 186 × 4 bytes)
Features (cached):  ~50 MB per dataset
Total:              ~1.4 GB
```

---

## Extending the Code

### Adding a New Dataset

1. **Create dataset wrapper** in `scenario_datasets/new_dataset.py`:
```python
class NewDataset(DatasetBase):
    def __init__(self, root_path, shots):
        super().__init__(root_path, shots)
        self.dataset_name = "new_dataset"
        self.read_data()
```

2. **Register in** `scenario_datasets/__init__.py`:
```python
dataset_list = {
    # ...
    "new_dataset": NewDataset,
}
```

3. **Add to config** `configs/test_config.yaml`:
```yaml
datasets: ["dtd", "oxford_flowers", "new_dataset"]
```

---

### Modifying the Algorithm

**Change fusion mechanism:**
```python
# Current: Linear fusion
final = (1 - α) * zeroshot + α * adapter

# Alternative: Learned fusion
fusion_weights = nn.Parameter(torch.ones(num_classes) * 0.8)
final = (1 - fusion_weights) * zeroshot + fusion_weights * adapter
```

**Add task-specific projections:**
```python
# Instead of shared Rand_Proj
self.task_projections = nn.ModuleList([
    nn.Linear(512, hidden_dim) for _ in range(num_tasks)
])
```

---

## References

### Papers

1. **CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
2. **Analytic Learning:** Zhuang et al., "ACIL: Analytic Class-Incremental Learning with Absolute Memorization", NeurIPS 2021
3. **Ridge Regression:** Hoerl & Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", Technometrics 1970

### Code Structure

```
Main Script (primal_RAIL.py)
    ↓
Configuration (configs/)
    ↓
Dataset Loading (scenario_datasets/)
    ↓
CLIP Model (clip/)
    ↓
Feature Extraction (utils.py)
    ↓
Analytic Learning (primal_RAIL.py)
```

---

## Summary

### Key Takeaways

1. **Primal RAIL uses closed-form solutions** (no gradient descent)
2. **Recursive updates enable efficient continual learning**
3. **Fusion combines generalization (CLIP) with specialization (adapter)**
4. **Feature caching speeds up repeated experiments**
5. **Windows compatibility requires `if __name__ == '__main__'` and `num_workers=0`**

### Code Flow in One Sentence

> Load CLIP → Extract features → For each task: expand features, update adapter weights recursively, evaluate on all tasks seen so far → Report final accuracies.

---

**Document Version:** 1.0  
**Last Updated:** January 27, 2026  
**For Support:** See [README.md](README.md) for contact information
