# AI-Powered Cooking Assistant with Multi-Modal Explainability

**COMP0220 Deep Learning Coursework**  
**University College London**

---

## Abstract

This project presents an end-to-end AI cooking assistant system that integrates computer vision and natural language processing to detect ingredients in refrigerator images and generate personalized recipe recommendations. The system comprises three main components: (1) ingredient detection using YOLOv8 and classification using fine-tuned EfficientNet-B0 and ResNet50 models achieving 60.18% and 62.87% accuracy respectively on 316 food categories, (2) recipe generation using parameter-efficiently fine-tuned GPT-2 models, and (3) a novel multi-modal explainability framework combining Grad-CAM visual attention maps with LLM-generated natural language explanations. The system demonstrates practical applicability in reducing food waste while addressing key concerns in AI interpretability, sustainability, and ethical deployment.

---

## Project Objectives

1. Train and evaluate multiple deep learning architectures for ingredient classification
2. Implement object detection for ingredient localization in refrigerator images
3. Fine-tune large language models for recipe generation and conversational interaction
4. Develop a novel multi-modal explainability approach combining visual and linguistic explanations
5. Integrate components into a production-ready end-to-end pipeline
6. Analyze environmental impact and ethical considerations

---

## System Architecture

### Component Overview

**Vision Pipeline**:
- YOLOv8 object detection for ingredient localization
- EfficientNet-B0 / ResNet50 for ingredient classification
- Grad-CAM for visual explainability

**Language Pipeline**:
- LSTM baseline for comparison
- GPT-2 (355M parameters) fine-tuned with PEFT/LoRA for recipe generation
- GPT-2 fine-tuned for conversational explanations

**Integration Layer**:
- Confidence-based filtering
- Multi-ingredient aggregation
- Error handling and graceful degradation

---

## Key Results

### Vision Models

Trained on Food-316 dataset (316 classes, ~6,676 images):

| Model | Test Accuracy | F1 Score (Macro) | F1 Score (Weighted) | Parameters |
|-------|---------------|------------------|---------------------|------------|
| ResNet50 | **62.87%** | **0.5735** | **0.6032** | 24.2M |
| EfficientNet-B0 | 60.18% | 0.5276 | 0.5669 | 4.4M |
| Random Baseline | 0.32% | - | - | - |

**Performance Analysis**:
- ResNet50 achieves highest test accuracy (62.87%) and F1 scores despite having 5.5× more parameters
- EfficientNet-B0 provides competitive performance (60.18%) with 82% fewer parameters, offering better efficiency
- Both models achieve ~190× improvement over random baseline (0.32%), demonstrating effective learning
- Best validation accuracies: ResNet50 (61.68%), EfficientNet-B0 (59.73%)

### Language Models

| Model | Architecture | Trainable Parameters | Perplexity | Quality |
|-------|-------------|---------------------|------------|---------|
| LSTM Baseline | 2-layer LSTM | 15M | High | Poor |
| GPT-2 Recipe | gpt2-medium + LoRA | 0.35M (0.1%) | Low | Excellent |
| GPT-2 Conversational | gpt2-medium + LoRA | 0.35M (0.1%) | Low | Excellent |

**PEFT/LoRA Advantages**:
- 99.9% parameter reduction compared to full fine-tuning
- Faster training convergence (~3 hours vs ~12 hours)
- Reduced overfitting risk
- Modular adapter architecture enables task switching

### Complete Pipeline Performance

- **Detection Rate**: Average 14 ingredients detected per refrigerator image
- **Classification Confidence**: Mean 72% for detected ingredients (threshold: 30%)
- **Recipe Generation**: Contextually appropriate recipes with conversational formatting
- **Explainability**: Integrated visual and linguistic explanations for all predictions

---

## Technical Innovation

### Multi-Modal Explainability Framework

**Motivation**: Traditional explainability methods (e.g., Grad-CAM) provide visual attention maps but lack semantic interpretation accessible to non-expert users.

**Approach**: 
1. Generate Grad-CAM heatmaps showing spatial attention regions
2. Extract prediction confidence and class information
3. Pass visual and prediction data to fine-tuned conversational LLM
4. Generate natural language explanation of model reasoning

**Example Output**:
```
Visual: [Grad-CAM heatmap highlighting fibrous texture regions]

Explanation: "The model identified this ingredient as chicken with 94% 
confidence by focusing on the characteristic fibrous texture and pale 
coloration typical of cooked poultry. The attention pattern suggests 
the model has learned to distinguish chicken from visually similar 
proteins such as turkey or pork."
```

**Advantages**:
- Interpretable to non-technical users
- Facilitates model debugging and bias detection
- Enhances user trust through transparency
- Enables educational applications

---

## Methodology

### Data Preparation

**Vision Dataset**:
- Source: Food-316 (Hugging Face: `Scuccorese/food-ingredients-dataset`)
- Preprocessing: Resizing (224×224), normalization, data augmentation
- Split: 80% train, 10% validation, 10% test

**Language Datasets**:
- Recipe data: Food.com (~50K recipes) + RecipeNLG (~2M recipes)
- Conversational data: OASST1 multi-turn dialogues
- Preprocessing: Cleaning, tokenization, special token insertion
- Format: Conversational with `<|user|>` and `<|assistant|>` markers

### Training Configuration

**Vision Models**:
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Batch size: 32
- Epochs: 20 with early stopping
- Regularization: L2, dropout (0.3), data augmentation
- Hardware: NVIDIA T4 GPU (Google Colab)

**Language Models**:
- Base: gpt2-medium (355M parameters)
- Method: PEFT/LoRA (r=8, α=16, dropout=0.1)
- Optimizer: AdamW (lr=2e-4)
- Batch size: 4 with gradient accumulation (effective=16)
- Epochs: 3
- Hardware: NVIDIA T4 GPU (Google Colab)

### Evaluation Metrics

**Vision**: Top-1/Top-5 accuracy, per-class precision/recall, confusion matrix analysis

**Language**: Perplexity, qualitative assessment, human evaluation of coherence and relevance

**System**: End-to-end success rate, ingredient detection accuracy, recipe appropriateness

---

## Notebook Organization

This submission includes 11 Jupyter notebooks organized into three categories:

### Vision Models (3 notebooks)
1. **Food Classifier Training**: EfficientNet-B0 and ResNet50 training and evaluation
2. **Grad-CAM Explainability**: Visual attention analysis and comparison
3. **YOLO Object Detection**: Ingredient detection and localization

### Language Models (6 notebooks)
1. **Data Preprocessing**: Dataset preparation and cleaning
2. **LSTM Baseline**: Baseline model for comparison
3. **GPT-2 Recipe Training**: Fine-tuning for recipe generation
4. **GPT-2 Conversational Training**: Fine-tuning for explanations
5. **Conversational Dataset Preparation**: Training data creation
6. **CookingBot Testing**: Model evaluation and testing

### System Integration (2 notebooks)
1. **Vision-LLM Integration**: Classifier and language model integration
2. **Complete Pipeline**: End-to-end system demonstration

Detailed descriptions and execution instructions are provided in `FILE_STRUCTURE.md`.

---

## Sustainability and Environmental Impact

### Carbon Footprint Analysis

**Training Emissions**:
- Vision models: ~0.3 kg CO₂ equivalent
- Language models: ~0.45 kg CO₂ equivalent
- Total: ~0.75 kg CO₂ equivalent

**Context**:
- Equivalent to charging a smartphone 15 times
- 0.00001% of GPT-3 training emissions (estimated 552 tons CO₂)
- Comparable to driving a car 3 kilometers

**Inference Emissions**: ~0.0001 kWh per recipe generation (negligible)

### Sustainability Strategies

1. **Model Efficiency**: EfficientNet architecture optimized for parameter efficiency
2. **Transfer Learning**: Leverages pre-trained models to reduce training requirements
3. **PEFT/LoRA**: Trains only 0.1% of parameters, reducing compute by 99.9%
4. **Edge Deployment Potential**: Models small enough for mobile device deployment

### Societal Impact

**Positive**:
- Reduces household food waste through ingredient utilization
- Increases accessibility of cooking knowledge
- Promotes sustainable consumption patterns

**Considerations**:
- Dataset bias toward Western cuisines may limit cultural applicability
- Requires internet connectivity and device access
- Energy consumption of inference at scale

---

## Ethical Considerations

### Data and Privacy
- All datasets are publicly available and appropriately licensed
- No personal data collection or user tracking
- No identifiable information in training data

### Bias and Fairness
- Food-316 dataset exhibits Western-centric bias in ingredient representation
- Recipe recommendations reflect training data distribution
- Explainability framework enables bias detection and mitigation

### Accessibility and Inclusion
- Natural language interface reduces technical barriers
- Conversational explanations accommodate diverse user backgrounds
- Potential for multi-lingual extension

### Responsible AI Practices
- Transparent model limitations communicated to users
- Confidence thresholds prevent low-quality predictions
- Open documentation enables reproducibility and scrutiny

---

## Limitations and Future Work

### Current Limitations
1. Classification accuracy constrained by dataset size and visual similarity between ingredients
2. Recipe generation limited to training data distribution
3. Explainability framework requires separate inference pass (increased latency)
4. System requires internet connectivity for model access

### Proposed Extensions
1. Expand training data with additional cuisines and ingredient categories
2. Implement retrieval-augmented generation for recipe diversity
3. Optimize explainability generation for real-time performance
4. Develop mobile application for edge deployment
5. Incorporate nutritional information and dietary restrictions
6. Multi-modal input (voice commands, barcode scanning)

---

## Reproducibility

All notebooks are self-contained and include:
- Dependency installation cells
- Path configuration
- Inline documentation
- Checkpoint saving
- Evaluation metrics

Execution instructions and environment setup are detailed in `FILE_STRUCTURE.md`.

**Estimated Reproduction Time**:
- With pre-trained models: ~30 minutes (demo)
- Full training pipeline: ~10 hours (GPU required)

**Hardware Requirements**:
- GPU: NVIDIA T4 or equivalent (Google Colab free tier sufficient)
- RAM: 15 GB minimum
- Storage: ~7 GB (Google Drive)

---

## Coursework Requirements Compliance

### Core Requirements
- ✓ Two image recognition models trained and evaluated
- ✓ Three dialogue models implemented (LSTM, GPT-2 Recipe, GPT-2 Conversational)
- ✓ Comprehensive ethical and environmental analysis
- ✓ Video/podcast documentation (20-30 minutes)

### Bonus Features
- ✓ Vision system integration with language models
- ✓ Advanced explainability approach (multi-modal XAI)
- ✓ Production-ready features (error handling, confidence thresholds)
- ✓ Novel contribution to interpretable AI

---

## Documentation

- **FILE_STRUCTURE.md**: Complete file organization and Google Drive setup
- **DATASET_REQUIREMENTS.md**: Dataset specifications and download instructions
- **SUBMISSION_CHECKLIST.md**: Final submission verification

---

## Acknowledgments

- YOLOv8 model training conducted by collaborator on refrigerator ingredient dataset
- Food-316 dataset provided by Scuccorese via Hugging Face
- Recipe data sourced from Food.com and RecipeNLG
- Conversational data from OASST1 (Open Assistant)
- Implementation built on PyTorch, Hugging Face Transformers, and timm libraries

---

## Declaration

This work represents original research conducted for COMP0220 Deep Learning coursework at University College London. All external sources and collaborations are appropriately acknowledged. The code, documentation, and analysis are the author's own work except where explicitly stated.

---

## References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI.
4. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
6. Jocher, G., et al. (2023). YOLOv8. Ultralytics.

---

**Submitted for COMP0220 Deep Learning**  
**University College London**  
**Academic Year 2024-2025**
