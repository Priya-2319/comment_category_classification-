# Comment Classification using Machine Learning

## Project Overview

This project is a **multi-label text classification system** that detects different categories of toxic comments using Natural Language Processing (NLP) and Machine Learning.

The project uses the **Jigsaw Toxic Comment Classification** dataset and predicts six categories of toxic behavior:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The text is represented using both **word-level and character-level TF-IDF features**. Three machine learning models were evaluated, and Logistic Regression was selected based on the validation results.

## Features

- Text cleaning and preprocessing
- Word-level TF-IDF using unigrams and bigrams
- Character-level TF-IDF using character n-grams
- Combination of word and character features
- Multi-label classification using One-vs-Rest
- Comparison of three machine learning models
- Kaggle submission with probability predictions

## Dataset

The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

The training dataset contains **159,571 comments** and six target labels.

| Label | Category |
|---|---|
| toxic | Toxic comment |
| severe_toxic | Severely toxic comment |
| obscene | Obscene content |
| threat | Threatening content |
| insult | Insulting content |
| identity_hate | Identity-based hate |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Regular Expressions
- TF-IDF
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes
- Kaggle

## Project Workflow

```text
Raw Comments
     ↓
Text Cleaning
     ↓
Train-Test Split
     ↓
Word TF-IDF ─────┐
                 ├──→ Combined Features
Character TF-IDF┘
                 ↓
       Multiple ML Models
                 ↓
        Model Comparison
                 ↓
      Logistic Regression
                 ↓
      Full Dataset Training
                 ↓
       Kaggle Predictions
```

## Text Preprocessing

The comments were cleaned before feature extraction. The preprocessing included:

- Converting text to lowercase
- Removing URLs
- Removing HTML tags
- Removing IP addresses
- Removing extra whitespace

## Feature Engineering

Two types of TF-IDF features were used.

### Word-Level TF-IDF

- Maximum features: 50,000
- N-gram range: `(1, 2)`

### Character-Level TF-IDF

- Maximum features: 50,000
- Character n-gram range: `(3, 5)`

The two feature matrices were combined to create a **100,000-feature representation**.

## Models Evaluated

Three machine learning algorithms were trained using the same validation data and combined TF-IDF features.

| Model | Macro ROC-AUC |
|---|---:|
| Logistic Regression | **0.98561** |
| Linear SVM | 0.97425 |
| Multinomial Naive Bayes | 0.92824 |

Logistic Regression achieved the highest validation Macro ROC-AUC among the three evaluated models and was selected for the final training stage.

## Final Results

The selected Logistic Regression model was retrained using all **159,571 labeled training comments**.

The final model generated probability predictions for the six toxicity categories on the Kaggle test dataset.

**Kaggle Private Leaderboard ROC-AUC: 0.97931**

The submission contained probability scores rather than binary predictions.

## Repository Contents

```text
├── README.md
├── submission.csv
└── notebook.ipynb
```

> File names may vary depending on the final repository structure.

## Future Scope

Possible future improvements include:

- Developing a Streamlit web interface
- Trying transformer-based NLP models such as BERT
- Improving handling of highly imbalanced labels
- Adding explainability for model predictions
- Comparing additional machine learning approaches

## Conclusion

This project demonstrates how NLP and machine learning can be used to classify online comments into multiple toxicity categories. Combining word-level and character-level TF-IDF features with Logistic Regression produced strong validation and Kaggle results.

## Dataset Source

The project uses the Jigsaw Toxic Comment Classification Challenge dataset provided on Kaggle.
