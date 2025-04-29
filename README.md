# Predicting Heart Disease Risk Using Ensemble and Deep Learning Models on Clinical Data

## Overview

This repository presents the implementation of a machine learning framework to predict the likelihood of heart disease in patients using structured clinical data. The project explores the efficacy of three machine learning models—Support Vector Machine (SVM), Random Forest, and Artificial Neural Network (ANN)—to classify the presence or absence of heart disease based on 14 clinical and demographic features. The analysis includes preprocessing, model development, evaluation, and clinical interpretation of results.

## Objectives

- To identify significant clinical features contributing to heart disease prediction.
- To implement and compare classification models: SVM, Random Forest, and ANN.
- To evaluate model performance using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- To interpret model output and identify the most influential predictors.
- To ensure ethical, fair, and reproducible data science practice.

## Dataset

- **Source**: Kaggle - UCI Heart Disease Dataset  
- **Description**: Includes 303 anonymised patient records with variables such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, heart rate, exercise-induced angina, ST depression, and ST slope.
- **Target Variable**: `HeartDisease` (1: presence, 0: absence)

## Methodology

### Preprocessing
- Removal of duplicates and handling of categorical variables using factorisation.
- Standardisation of numerical features via `StandardScaler`.
- Verification of missing values and data types.
- Class distribution assessment (balanced dataset).

### Modelling
Three classifiers were implemented:
1. **Support Vector Machine (SVM)**: RBF kernel, grid-searched hyperparameters.
2. **Random Forest**: Tuned via `GridSearchCV`, evaluated for feature importance.
3. **Artificial Neural Network (ANN)**: Built with TensorFlow/Keras using a multi-layer perceptron architecture.

### Evaluation Metrics
Models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

## Results

| Model           | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|----------|
| SVM            | 0.887    | 0.895     | 0.867  | 0.881    | 0.940    |
| Random Forest  | 0.907    | 0.934     | 0.867  | 0.900    | 0.949    |
| ANN            | 0.887    | 0.895     | 0.867  | 0.881    | 0.928    |

- **Random Forest** outperformed other models in most metrics and demonstrated the best generalisation.
- **Important features** (via Random Forest): `ST_Slope`, `Oldpeak`, `ExerciseAngina`, `ChestPainType`.
- **Confusion Matrix** showed Random Forest had the lowest false positives (6) and competitive recall.

## Tools & Technologies

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- TensorFlow / Keras (for ANN implementation)
- Jupyter Notebook
- Git and GitHub for version control and collaboration

## Ethical Considerations

- The dataset used is publicly available and fully anonymised.
- No personal data or human subjects were involved.
- Compliance with GDPR and University of Hertfordshire ethical guidelines was maintained.
- The analysis ensures fairness, transparency, and reproducibility.

## Project Structure
## How to Run

1. Clone the repository.
2. Install dependencies:
3. Open the Jupyter Notebook in the `notebooks/` directory to follow the complete workflow from data preprocessing to model evaluation.

## Author and Supervision

- **Student**: Somesh Nakka  
- **Student ID**: 23006089  
- **Supervisor**: Calum Morris  
- **Degree**: MSc Data Science  
- **University**: University of Hertfordshire  
- **Module Code**: 7PAM2002  
- **Submission Date**: 27 April 2025  

## License

This project is for academic and research purposes only. Dataset use complies with Kaggle’s data licensing terms.
