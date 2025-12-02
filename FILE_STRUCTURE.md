# File Structure and Setup Guide

## Submission Package Structure

```
COMP0220_Coursework_Final/
│
├── README.md                               
├── FILE_STRUCTURE.md                       
├── SUBMISSION_CHECKLIST.md                 
├── DATASET_REQUIREMENTS.md                 
│
├── notebooks/
│   ├── datasets/
│   │   └── datasets/
│   │       ├── Cleaned/
│   │       │   ├── clean_recipes.csv
│   │       │   ├── conversational_training_data.csv
│   │       │   ├── nutrition_lookup.csv
│   │       │   └── recipe_gpt2/
│   │       │       ├── train/
│   │       │       │   ├── data-00000-of-00001.arrow
│   │       │       │   ├── dataset_info.json
│   │       │       │   └── state.json
│   │       │       └── val/
│   │       │           ├── data-00000-of-00001.arrow
│   │       │           ├── dataset_info.json
│   │       │           └── state.json
│   │       ├── OASST1/
│   │       │   └── processed/
│   │       │       └── oasst1_multiturn_en/
│   │       │           ├── samples.txt
│   │       │           ├── train/
│   │       │           │   ├── data-00000-of-00001.arrow
│   │       │           │   ├── dataset_info.json
│   │       │           │   └── state.json
│   │       │           └── val/
│   │       │               ├── data-00000-of-00001.arrow
│   │       │               ├── dataset_info.json
│   │       │               └── state.json
│   │       ├── kaggleFood/
│   │       │   ├── RAW_recipes.csv
│   │       │   ├── RAW_interactions.csv
│   │       │   └── [other Food.com files]
│   │       └── recipeNLG/
│   │           └── RecipeNLG_dataset.csv
│   │
│   ├── vision/                             
│   │   ├── 01_Food_Classifier_Training.ipynb
│   │   ├── 02_GradCAM_Explainability.ipynb
│   │   └── 03_YOLO_Object_Detection.ipynb
│   │
│   ├── llm/                                
│   │   ├── 01_Data_Preprocessing.ipynb
│   │   ├── 02_LSTM_Baseline.ipynb
│   │   ├── 03_GPT2_Recipe_Training.ipynb
│   │   ├── 04_GPT2_Conversational_Training.ipynb
│   │   ├── 05_Conversational_Dataset_Prep.ipynb
│   │   └── 06_CookingBot_Testing.ipynb
│   │
│   └── integration/                        
│       ├── 01_Vision_LLM_Integration.ipynb
│       └── 02_Complete_Fridge_to_Recipe_Pipeline.ipynb
│
└── documentation/
    └── [additional documentation if needed]
```

---

## Google Drive Structure (For Running Notebooks)

The notebooks expect the following structure in Google Drive:

```
/content/drive/MyDrive/LLM_Models/
│
├── datasets/
│   └── datasets/                           # Dataset files (auto-downloaded via gdown)
│       ├── Cleaned/
│       ├── OASST1/
│       ├── kaggleFood/
│       └── recipeNLG/
│
├── cooking-assistant-project/
│   └── models/                             
│       ├── efficientnet_best.pth          # 21 MB
│       ├── resnet_best.pth                # 98 MB
│       ├── label_to_ingredient.json       
│       ├── ingredient_to_label.json       
│       ├── test_indices.pkl               
│       │
│       ├── lstm_dummy/                    
│       │   ├── model.pth
│       │   └── config.json
│       │
│       ├── yolo/                          
│       │   └── best.pt                    
│       │
│       └── gpt2-conversational-v1/        
│           └── final/
│               ├── adapter_model.safetensors
│               ├── adapter_config.json
│               ├── config.json
│               ├── tokenizer.json
│               ├── tokenizer_config.json
│               ├── vocab.json
│               ├── merges.txt
│               └── special_tokens_map.json
│
└── models/
    └── gpt2-recipe-final/                  
        └── final/
            ├── adapter_model.safetensors
            ├── adapter_config.json
            ├── config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── vocab.json
            ├── merges.txt
            └── special_tokens_map.json
```

---

## Dataset Information

### Included in Submission Package

The `notebooks/datasets/datasets/` directory contains all preprocessed datasets required for training:

| Dataset | Size | Purpose |
|---------|------|---------|
| `Cleaned/recipe_gpt2/` | ~87 MB | GPT-2 recipe model training data |
| `OASST1/processed/` | ~15 KB | GPT-2 conversational model training data |
| `Cleaned/*.csv` | ~99 MB | Preprocessed recipe and nutrition data |
| `kaggleFood/` | ~4 GB | Raw Food.com dataset (for preprocessing) |
| `recipeNLG/` | ~500 MB | Raw RecipeNLG dataset (for preprocessing) |

### Auto-Downloaded Datasets

The Food-316 vision dataset (~2.5 GB) is automatically downloaded from Hugging Face (`Scuccorese/food-ingredients-dataset`) on first execution of vision notebooks.

---

## Model Files

### Vision Models

**Location**: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/`

| File | Size | Description |
|------|------|-------------|
| `efficientnet_best.pth` | 21 MB | EfficientNet-B0 trained weights |
| `resnet_best.pth` | 98 MB | ResNet50 trained weights |
| `label_to_ingredient.json` | 15 KB | Class index to ingredient name mapping |
| `ingredient_to_label.json` | 15 KB | Ingredient name to class index mapping |
| `test_indices.pkl` | 50 KB | Test set split indices |

### YOLO Model

**Location**: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/yolo/`

| File | Size | Description |
|------|------|-------------|
| `best.pt` | 50-100 MB | YOLOv8 trained for ingredient detection |

### Language Models

**Recipe Model**: `/content/drive/MyDrive/LLM_Models/models/gpt2-recipe-final/final/`

**Conversational Model**: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/gpt2-conversational-v1/final/`

| File | Size | Description |
|------|------|-------------|
| `adapter_model.safetensors` | 16.5 MB | PEFT/LoRA adapter weights |
| `config.json` | 1 KB | Model configuration |
| `tokenizer.json` | 2 MB | Tokenizer vocabulary |
| `vocab.json` | 779 KB | Vocabulary mappings |
| `merges.txt` | 446 KB | BPE merge rules |
| Other files | <1 KB each | Tokenizer configuration |

### LSTM Baseline

**Location**: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/lstm_dummy/`

| File | Size | Description |
|------|------|-------------|
| `model.pth` | 5 MB | LSTM model weights |
| `config.json` | <1 KB | Model configuration |

---

## Notebook Execution Order

### Recommended Execution Sequence

**Phase 1: Data Preprocessing** (~1 hour)
1. `llm/01_Data_Preprocessing.ipynb` - Preprocess raw recipe datasets
2. `llm/05_Conversational_Dataset_Prep.ipynb` - Generate conversational training data

**Phase 2: Vision Models** (~3 hours)
1. `vision/01_Food_Classifier_Training.ipynb` - Train EfficientNet-B0 and ResNet50
2. `vision/02_GradCAM_Explainability.ipynb` - Generate Grad-CAM visualizations
3. `vision/03_YOLO_Object_Detection.ipynb` - Test YOLO object detection

**Phase 3: Language Models** (~6 hours)
1. `llm/02_LSTM_Baseline.ipynb` - Train LSTM baseline
2. `llm/03_GPT2_Recipe_Training.ipynb` - Fine-tune GPT-2 for recipe generation
3. `llm/04_GPT2_Conversational_Training.ipynb` - Fine-tune GPT-2 for conversation
4. `llm/06_CookingBot_Testing.ipynb` - Evaluate language models

**Phase 4: System Integration** (~1 hour)
1. `integration/01_Vision_LLM_Integration.ipynb` - Integrate vision and language models
2. `integration/02_Complete_Fridge_to_Recipe_Pipeline.ipynb` - Complete end-to-end system

### Quick Demo Execution

For demonstration purposes with pre-trained models:
1. `integration/02_Complete_Fridge_to_Recipe_Pipeline.ipynb` - Complete system demo
2. `vision/02_GradCAM_Explainability.ipynb` - Explainability analysis

---

## Setup Requirements

### Google Colab Environment
- Python 3.10+
- CUDA-enabled GPU (recommended for training)
- 15 GB RAM minimum
- 50 GB disk space

### Google Drive Storage
- Models: ~1 GB
- Datasets: ~5 GB (if including raw data)
- Outputs: ~500 MB
- Total: ~6.5 GB

### External Dependencies

All notebooks include installation cells for required packages:
- PyTorch 2.0+
- Transformers (Hugging Face)
- timm (PyTorch Image Models)
- pytorch-grad-cam
- ultralytics (YOLO)
- datasets (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)

---

## Path Verification

Execute the following in a Colab cell after mounting Google Drive to verify correct setup:

```python
import os

BASE_DIR = "/content/drive/MyDrive/LLM_Models/cooking-assistant-project"
MODEL_DIR = f"{BASE_DIR}/models"

print("Vision Models:")
print(f"  EfficientNet: {os.path.exists(f'{MODEL_DIR}/efficientnet_best.pth')}")
print(f"  ResNet: {os.path.exists(f'{MODEL_DIR}/resnet_best.pth')}")
print(f"  YOLO: {os.path.exists(f'{MODEL_DIR}/yolo/best.pt')}")

print("\nLanguage Models:")
print(f"  Recipe: {os.path.exists('/content/drive/MyDrive/LLM_Models/models/gpt2-recipe-final/final')}")
print(f"  Conversational: {os.path.exists(f'{MODEL_DIR}/gpt2-conversational-v1/final')}")

print("\nDatasets:")
print(f"  Recipe GPT-2: {os.path.exists('/content/datasets/datasets/Cleaned/recipe_gpt2')}")
print(f"  OASST1: {os.path.exists('/content/datasets/datasets/OASST1/processed')}")
```

Expected output: All paths return `True`.

---

## Dataset Auto-Download

Notebooks include automatic dataset download functionality using `gdown`:

```python
import gdown
folder_id = "1HiAxfpV-auZECGKufjhgZryg9BhA0RjM"
gdown.download_folder(id=folder_id, output="/content/datasets", quiet=False)
```

This downloads all required preprocessed datasets (~186 MB) from Google Drive on first execution.

---

## Storage Requirements Summary

| Component | Size | Location |
|-----------|------|----------|
| Vision models | ~120 MB | Google Drive |
| Language models | ~35 MB | Google Drive |
| YOLO model | ~75 MB | Google Drive |
| Preprocessed datasets | ~186 MB | Auto-download |
| Raw datasets | ~4.5 GB | Optional (for preprocessing) |
| Food-316 dataset | ~2.5 GB | Auto-download (Hugging Face) |
| **Total (minimal)** | **~3 GB** | **Colab + Drive** |
| **Total (with raw data)** | **~7.5 GB** | **Colab + Drive** |

---

## File Naming Conventions

All file names follow lowercase with underscores convention:
- Model files: `{model_name}_best.pth`
- Config files: `{descriptor}.json`
- Dataset files: `{dataset_name}_{split}.{ext}`

Case-sensitive paths must match exactly as specified in this document.

---

## Troubleshooting

### Path Not Found Errors
Verify Google Drive is mounted: `/content/drive/MyDrive/` should exist.
Check file paths match the structure specified above.

### Model Loading Failures
Ensure model files are not corrupted (verify file sizes).
Re-upload model files if necessary.

### Dataset Download Issues
Check internet connectivity in Colab environment.
Verify Google Drive folder permissions are set to "Anyone with link can view".

### Out of Memory Errors
Reduce batch size in training notebooks.
Use Colab Pro for increased RAM allocation.
Clear cached datasets: `!rm -rf ~/.cache/huggingface/`

---

## Submission Contents

This submission package includes:
- 11 Jupyter notebooks (organized by category)
- Complete preprocessed datasets
- Documentation files
- File structure guide (this document)

Model weight files are stored separately in Google Drive due to size constraints.

---

## Additional Notes

All notebooks are self-contained and include:
- Dependency installation cells
- Path configuration cells
- Inline documentation
- Output visualization cells
- Model checkpoint saving

Notebooks can be executed independently or in sequence as specified in the execution order section.
