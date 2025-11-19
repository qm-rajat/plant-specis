# Plant Species Classification Project TODO

## 1. Set up project structure
- [x] Create directories: src/, models/
- [x] Create requirements.txt with dependencies (tensorflow, streamlit, pillow, numpy, matplotlib, scikit-learn)

## 2. Data preparation
- [x] Load images from data/corn/, data/grape/, data/non_leaf/
- [x] Preprocess images: resize to 224x224, normalize
- [x] Split data into train/val/test (70/20/10) for leaf detector and species classifier

## 3. Build leaf detector
- [x] Use pre-trained MobileNetV2 from TensorFlow Hub
- [x] Fine-tune on leaf vs non-leaf data (binary classification)

## 4. Build species classifier
- [x] Build CNN model for corn vs grape classification
- [x] Train on corn and grape images

## 5. Train models
- [x] Train leaf detector and save to models/
- [x] Train species classifier and save to models/
- [x] Implement logic for unknown species (low confidence)

## 6. Build Streamlit app
- [x] Create app.py: upload image, run predictions, display results
- [x] Integrate two-stage classification: leaf check then species

## 7. Test app locally
- [x] Install dependencies
- [x] Run data prep and training scripts
- [x] Launch Streamlit app and test with sample images
