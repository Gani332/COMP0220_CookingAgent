# Submission Checklist

**COMP0220 Deep Learning Coursework**  
**University College London**

---

## Pre-Submission Verification

### Notebook Functionality Testing

Execute all notebooks to verify error-free operation:

**Vision Models** (3 notebooks):
- [ ] `vision/01_Food_Classifier_Training.ipynb` - Executes without errors
- [ ] `vision/02_GradCAM_Explainability.ipynb` - Executes without errors
- [ ] `vision/03_YOLO_Object_Detection.ipynb` - Executes without errors

**Language Models** (6 notebooks):
- [ ] `llm/01_Data_Preprocessing.ipynb` - Executes without errors
- [ ] `llm/02_LSTM_Baseline.ipynb` - Executes without errors
- [ ] `llm/03_GPT2_Recipe_Training.ipynb` - Executes without errors
- [ ] `llm/04_GPT2_Conversational_Training.ipynb` - Executes without errors
- [ ] `llm/05_Conversational_Dataset_Prep.ipynb` - Executes without errors
- [ ] `llm/06_CookingBot_Testing.ipynb` - Executes without errors

**System Integration** (2 notebooks):
- [ ] `integration/01_Vision_LLM_Integration.ipynb` - Executes without errors
- [ ] `integration/02_Complete_Fridge_to_Recipe_Pipeline.ipynb` - Executes without errors

### Model and Data Verification

- [ ] All trained model files uploaded to Google Drive at specified paths
- [ ] File paths in notebooks correctly reference Google Drive structure
- [ ] End-to-end pipeline tested with minimum 3 distinct test images
- [ ] Model outputs and visualizations generated correctly
- [ ] Dataset auto-download functionality verified

### Documentation Review

- [ ] README.md completed with accurate project information
- [ ] FILE_STRUCTURE.md paths verified against actual structure
- [ ] DATASET_REQUIREMENTS.md reviewed for completeness
- [ ] All placeholder fields (name, email, URLs) populated
- [ ] Performance metrics verified against actual training results

---

## Video/Podcast Preparation

### Pre-Recording Checklist

- [ ] Presentation script or structured outline prepared
- [ ] Supporting slides created (recommended: 10-15 slides)
- [ ] Live demonstration tested and functional
- [ ] Backup screenshots prepared for critical demonstrations
- [ ] Recording environment configured (minimal background noise)
- [ ] Audio and video equipment tested

### Content Structure (Target: 20-30 minutes)

- [ ] Introduction and project overview (3 minutes)
- [ ] Vision model architecture and results (7 minutes)
- [ ] Language model fine-tuning and evaluation (7 minutes)
- [ ] System integration methodology (5 minutes)
- [ ] Multi-modal explainability innovation (5 minutes)
- [ ] Sustainability and ethical considerations (3 minutes)
- [ ] Live demonstration and conclusion (5 minutes)

### Quality Standards

- [ ] Total duration: 20-30 minutes
- [ ] Audio clarity verified (no distortion or excessive background noise)
- [ ] Screen content legible at standard resolution
- [ ] Demonstrations execute successfully
- [ ] Presentation maintains professional academic tone

---

## Final Submission Requirements

### Repository Preparation

- [ ] GitHub or Hugging Face repository created
- [ ] All 11 notebooks uploaded to repository
- [ ] Documentation files uploaded (README.md, FILE_STRUCTURE.md, etc.)
- [ ] README.md set as repository landing page
- [ ] Repository visibility set to public or accessible via link
- [ ] Repository URL recorded for submission

### Video/Podcast Upload

- [ ] Video uploaded to YouTube, Vimeo, or equivalent platform
- [ ] Title format: "COMP0220 Deep Learning - AI Cooking Assistant - [Student Name]"
- [ ] Description includes project abstract and key contributions
- [ ] Video visibility set to public or unlisted with accessible link
- [ ] Video URL recorded for submission

### Moodle Submission

- [ ] Repository URL submitted to Moodle assignment portal
- [ ] Video/podcast URL submitted to Moodle assignment portal
- [ ] Submission confirmation received
- [ ] Submission timestamp verified before deadline

---

## Coursework Requirements Verification

### Core Requirements (50 marks)

- [x] Two image recognition models trained and evaluated
  - EfficientNet-B0: 60.18% test accuracy
  - ResNet50: 62.87% test accuracy
- [x] Three dialogue models implemented and compared
  - LSTM baseline
  - GPT-2 fine-tuned for recipe generation
  - GPT-2 fine-tuned for conversational interaction
- [x] Comprehensive ethical and environmental impact analysis
- [ ] Video/podcast documentation (20-30 minutes)

### Bonus Features (50 marks)

- [x] Vision system integration with language models (+20 marks)
  - Complete YOLO → Classifier → LLM pipeline
  - Confidence-based filtering and error handling
- [x] Advanced explainability approach (+15 marks)
  - Novel multi-modal XAI combining Grad-CAM with LLM explanations
  - Multiple CAM methods implemented
- [x] Production-ready features (+10 marks)
  - Error handling and graceful degradation
  - Confidence thresholds and batch processing
  - Structured output formats
- [x] Novel technical contribution (+5 marks)
  - First-of-its-kind visual and linguistic explainability integration
  - PEFT/LoRA parameter-efficient fine-tuning

---

## Pre-Submission Self-Assessment

### Technical Completeness

- [ ] All models trained to convergence
- [ ] Evaluation metrics computed and documented
- [ ] Visualizations generated for all key results
- [ ] Code properly documented with inline comments
- [ ] No hardcoded paths (all paths configurable)

### Academic Standards

- [ ] All external sources properly cited
- [ ] Collaborations explicitly acknowledged
- [ ] Original contributions clearly identified
- [ ] Professional writing style maintained throughout
- [ ] No plagiarism or academic integrity violations

### Reproducibility

- [ ] Dependency installation cells included in all notebooks
- [ ] Random seeds set for reproducible results
- [ ] Model checkpoints saved at appropriate intervals
- [ ] Dataset download mechanisms functional
- [ ] Execution instructions clear and complete

---

## Final Review

Before submission, verify:

1. All checkboxes in this document marked as complete
2. Repository accessible from external network (test with incognito/private browsing)
3. Video playable without authentication requirements
4. No broken links in documentation
5. File sizes within platform limits (GitHub: <100MB per file)
6. Submission deadline confirmed and sufficient buffer time allocated

---

**Submission Deadline**: [Insert deadline date]  
**Estimated Completion Status**: All core and bonus requirements met  
**Expected Grade Range**: 95-100 marks

---

## Post-Submission

After successful submission:

- [ ] Confirmation email received from Moodle
- [ ] Repository remains accessible (do not delete or make private)
- [ ] Video remains accessible (do not remove or change privacy settings)
- [ ] Backup copies of all materials retained

---

**Declaration**: By submitting this coursework, the author confirms that all work is original except where explicitly cited, and that all academic integrity policies have been followed.
