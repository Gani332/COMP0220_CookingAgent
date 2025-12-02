# Remaining Tasks for COMP0220 Coursework

**Remaining**: 3 Tasks (2 Critical, 1 Bonus)

---

## Task 1: AI Literacy Explainability Integration 

### Requirement

The podcast must demonstrate how the chatbot explains CNN decisions to make AI literacy accessible to the target audience.

### Current Status

- ✓ Grad-CAM implementation exists (notebook 02_GradCAM_Explainability.ipynb)

- ✓ Conversational LLM trained (GPT-2 conversational model)

- ✗ NOT integrated together

### What Needs to Be Done

**Create New Notebook**: `notebooks/integration/03_AI_Literacy_Explainability.ipynb`

**Implementation Steps**:

1. **Load Required Models**
  - Vision classifier (EfficientNet or ResNet)
  - Grad-CAM module
  - Conversational GPT-2 model

1. **Create Explainability Pipeline**

   ```python
   # Pseudo-code structure
   def explain_prediction(image):
       # Step 1: Get prediction
       prediction, confidence = classifier.predict(image)
       
       # Step 2: Generate Grad-CAM heatmap
       heatmap = gradcam.generate(image, prediction)
       
       # Step 3: Create explanation prompt
       prompt = f"<|user|> Why did the model classify this as {prediction} with {confidence}% confidence? Explain how neural networks learn to recognize food. <|assistant|>"
       
       # Step 4: Generate natural language explanation
       explanation = conversational_model.generate(prompt)
       
       # Step 5: Return combined visual + text
       return {
           'image': image,
           'heatmap': heatmap,
           'prediction': prediction,
           'confidence': confidence,
           'explanation': explanation
       }
   ```

1. **AI Literacy Questions to Implement**
  - "How does a computer learn to recognize food?"
  - "Why did the model classify this as chicken?"
  - "What features does the network look at?"
  - "How can I explain neural networks to my grandmother?"

1. **Create Demo Examples**
  - 3-5 food images with full explanations
  - Show correct and incorrect predictions
  - Demonstrate visual attention + linguistic explanation

### Deliverable

- Notebook with working integration

- 3-5 demo examples ready for podcast

- Screenshots/outputs saved for presentation



---

## Task 2: Sustainability Discussion with LLM 

### Requirement

Section 4 of podcast must include sustainability metric evaluation with LLM component explaining environmental trade-offs.

### Current Status

- ✓ Sustainability calculations done (CO₂: ~0.75 kg, energy: ~1.5 kWh)

- ✓ Analysis in README.md

- ✗ LLM cannot discuss sustainability interactively

### What Needs to Be Done

**Option A: Add to Existing Notebook** (`06_CookingBot_Testing.ipynb`)

**Option B: Create New Section** in `03_AI_Literacy_Explainability.ipynb`

**Implementation Steps**:

1. **Prepare Sustainability Context**

   ```python
   sustainability_context = """
   Training Statistics:
   - Vision models: ~0.3 kg CO₂ (~0.6 kWh)
   - Language models: ~0.45 kg CO₂ (~0.9 kWh)
   - Total: ~0.75 kg CO₂ (~1.5 kWh)
   - Equivalent to charging phone 15 times
   - Comparable to driving car 3 km
   
   Benefits:
   - Reduces household food waste
   - Helps users utilize ingredients before expiration
   - Promotes sustainable consumption
   """
   ```

1. **Create Sustainability Q&A System**

   ```python
   sustainability_questions = [
       "What is the environmental impact of training these AI models?",
       "How does this system help reduce food waste?",
       "Is the energy cost of AI worth the environmental benefits?",
       "How can we make AI more sustainable?",
       "Compare the carbon footprint of this system to training GPT-3"
   ]
   
   for question in sustainability_questions:
       prompt = f"<|user|> {sustainability_context}\n\n{question} <|assistant|>"
       response = conversational_model.generate(prompt)
       print(f"Q: {question}")
       print(f"A: {response}\n")
   ```

1. **Demonstrate Trade-offs**
  - Energy consumption vs food waste reduction
  - Training cost vs inference efficiency
  - Model size vs accuracy vs sustainability

1. **Create Visualization**
  - Bar chart comparing CO₂ emissions (this project vs GPT-3 vs daily activities)
  - Trade-off diagram (accuracy vs energy consumption)

### Deliverable

- Interactive sustainability Q&A with chatbot

- 5-7 key questions answered by LLM

- Visualizations for podcast

- Saved outputs/screenshots



---

## Task 3: User Evaluation (BONUS)

### Requirement

Test chatbot with ~5 real users, collect feedback using satisfaction survey, compare user enjoyment vs benchmark performance.

### Current Status

- ✗ Not done

### What Needs to Be Done

**Create User Study**:

1. **Prepare Test Environment**
  - Deploy chatbot in accessible format (Colab notebook or simple interface)
    - Prepare 3-5 test scenarios:
      - Recipe generation from ingredients
      - Ingredient identification from image
      - Conversational Q&A about cooking
      - AI literacy explanation request

1. **Recruit 5 Users**
  - Friends, family, classmates
  - Mix of technical and non-technical backgrounds
  - 10-15 minutes per user

1. **User Satisfaction Survey** (from coursework brief appendix)

   ```
   1. How enjoyable was your interaction with the chatbot today?
      □ Very enjoyable  □ Enjoyable  □ Neutral  □ Not very enjoyable  □ Not enjoyable at all
   
   2. Did the chatbot understand your questions and respond appropriately?
      □ Always  □ Most of the time  □ Sometimes  □ Rarely  □ Never
   
   3. How engaging did you find the chatbot's tone and personality?
      □ Very engaging  □ Somewhat engaging  □ Neutral  □ Not very engaging  □ Not engaging at all
   
   4. Would you choose to interact with this chatbot again in the future?
      □ Definitely  □ Probably  □ Not sure  □ Probably not  □ Definitely not
   
   5. What could be improved to make your experience more enjoyable?
      [Open-ended response]
   ```

1. **Collect Metrics**
  - Session length (longer = more engagement)
  - Number of interactions per user
  - Task success rate
  - Qualitative feedback

1. **Analysis**
  - Calculate average satisfaction scores
  - Compare user enjoyment vs perplexity scores
  - Identify common pain points
  - Quote interesting user feedback

1. **Create Results Summary**
  - Table of user ratings
  - Key insights
  - Comparison: User satisfaction vs benchmark performance
  - Recommendations for improvement

### Deliverable

- Survey responses from 5 users

- Analysis notebook or document

- Summary for podcast (1-2 slides)

- Quotes/testimonials

### Time Estimate

4-6 hours (including user recruitment and testing)

### Priority

**Optional but highly recommended** - Worth significant bonus marks and demonstrates real-world validation.

---

## Summary Checklist

### Critical (Must Do)

- [ ] Task 1: AI Literacy Explainability Integration (3-4 hours)

   - [ ] Create integration notebook

   - [ ] Implement Grad-CAM + LLM pipeline

   - [ ] Generate 3-5 demo examples

   - [ ] Save outputs for podcast

- [ ] Task 2: Sustainability Discussion with LLM (2-3 hours)

   - [ ] Add sustainability Q&A to chatbot

   - [ ] Test 5-7 key questions

   - [ ] Create visualizations

   - [ ] Save outputs for podcast

### Bonus (Recommended)

- [ ] Task 3: User Evaluation (4-6 hours)

   - [ ] Recruit 5 users

   - [ ] Conduct user testing

   - [ ] Collect survey responses

   - [ ] Analyze results

   - [ ] Create summary for podcast

---

## Podcast Integration

Once tasks complete, podcast Section 3 and 4 should include:

**Section 3: Application Demos**

- Show AI literacy explainability demo

- Display Grad-CAM + LLM explanation

- Answer "How does a computer learn?"

**Section 4: Sustainability Discussion**

- Present sustainability metrics

- Demo chatbot answering sustainability questions

- Show trade-off visualizations

- Discuss ethical implications

**Optional: User Evaluation Results**

- Present user satisfaction scores

- Share interesting user quotes

- Compare user enjoyment vs technical benchmarks

---

## Technical Notes

### Models Needed

- Vision classifier: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/efficientnet_best.pth`

- Conversational LLM: `/content/drive/MyDrive/LLM_Models/cooking-assistant-project/models/gpt2-conversational-v1/final/`

- Grad-CAM: Already implemented in notebook 02

### Key Libraries

```python
pip install pytorch-grad-cam transformers peft torch torchvision pillow matplotlib
```

### Prompt Format for Conversational Model

```python
prompt = "<|user|> [YOUR QUESTION] <|assistant|>"
```

