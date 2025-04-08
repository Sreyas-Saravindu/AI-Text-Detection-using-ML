# AI-Text-Detection-using-ML
This study presents an advanced AI text detection model that combines BERT embeddings with publication metadata to distinguish between human-written and AI-generated content. The model achieves impressive accuracy rates of 96% on validation data and 94% on test data, demonstrating robust performance in identifying AI-generated text.
# Dataset Link: https://docs.google.com/spreadsheets/d/1Kk7ZbGFQLxGdaySDvagHUdvLbYgeKWO0wm2Ibk93XHk/edit?usp=sharing

# Model Architecture and Features
The model employs a hybrid approach that integrates multiple feature types to enhance detection capabilities. At its core, the system utilizes BERT (Bidirectional Encoder Representations from Transformers) to extract semantic embeddings from text samples. These embeddings capture the linguistic patterns and contextual relationships within the content. The model augments these text-based features with publication metadata, specifically author h-index proxies and journal impact factors, which serve as indicators of scholarly credibility. Additionally, basic text statistics such as word count provide supplementary signals for classification.

# Implementation Details
The implementation follows a structured pipeline approach. First, the system creates publication metrics by analyzing patterns across the dataset, calculating proxy h-indices based on publication frequency and volume. Next, BERT embeddings are extracted from text samples using the CLS token representation. These features are combined with the publication metrics and fed into a machine learning pipeline consisting of a StandardScaler for normalization and a LogisticRegression classifier with balanced class weights to account for potential dataset imbalances.

# Performance Evaluation
The model demonstrates excellent performance across multiple evaluation metrics. On the test dataset, it achieves 94% accuracy with balanced precision and recall scores for both human and AI-generated content classes. The confusion matrix reveals strong classification capabilities with minimal misclassifications. The ROC curve analysis further confirms the model's discriminative power with a high AUC score, indicating its ability to effectively separate the two classes across different classification thresholds.

# Practical Applications
The system provides a practical solution for detecting AI-generated content in various contexts, particularly in academic and publishing environments where maintaining content authenticity is crucial. When tested on unseen examples, the model consistently identifies AI-generated text with high confidence scores, demonstrating its potential utility as a screening tool for editors, reviewers, and content managers. The inclusion of publication metadata enhances the model's contextual awareness, making it particularly effective for scholarly content assessment.
