# predicting-antibiotic-resistance

# Predicting Antibiotic Resistance with Machine Learning

## Overview
This project predicts antimicrobial resistance (AMR) using hospital microbiology lab data. I applied machine learning techniques to classify whether bacterial isolates are resistant or susceptible to antibiotics, supporting better empiric therapy decisions in healthcare.

## Problem Statement
Antibiotic resistance is a major public health threat, making infections harder to treat. Resistance patterns vary by bacterial species, specimen source, and hospital location. The goal of this project is to build a predictive model that helps clinicians anticipate resistance, improving treatment choices and stewardship.

## Dataset
- **Source:** [DRIAMS (Database for Resistance against Antimicrobials)](https://www.kaggle.com/datasets/drscarlat/driams)
- **Description:** Anonymized lab test results from hospitals in Switzerland, including:
  - Bacterial species
  - Antibiotic tested
  - Susceptibility result (S/R)
  - Specimen source
  - Hospital site
  - Year of test

## Tools and Technologies
- **Python** (pandas, scikit-learn, XGBoost)
- **Seaborn/Matplotlib** for visualizations
- **Jupyter Notebook** for exploratory data analysis (EDA)
- `.py` scripts for modular data cleaning and modeling pipelines

## Methodology
1. **Data Cleaning**
   - Removed missing/duplicate rows
   - Standardized categorical values
   - Filtered to susceptible/resistant outcomes

2. **Feature Engineering**
   - One-hot encoded categorical features:
     - Bacterial species
     - Antibiotic
     - Specimen source
     - Hospital site

3. **Modeling**
   - Used **XGBoost** classifier
   - Target: Resistant (1) vs. Susceptible (0)
   - Train-test split with stratification

4. **Evaluation**
   - Accuracy ~88%
   - Classification report (Precision/Recall/F1)
   - Confusion matrix
   - Feature importance analysis

## Visualizations
- Feature importance chart (top predictors)
- Resistance rate by species, antibiotic, and specimen source
- Heatmap of species-antibiotic resistance patterns
- Stratified bar chart by specimen source
- Model performance metrics (Precision, Recall, F1-score by class)

*(Example images saved in the /plots folder)*

## Key Findings
- Certain species (e.g., *Staphylococcus epidermidis*, *Proteus mirabilis*) showed higher resistance rates.
- Resistance patterns varied significantly by specimen source (e.g., blood vs. urine).
- Some antibiotics had consistently higher predicted resistance rates, aiding empiric selection.

## Future Work
- Include patient-level data (age, diagnosis) for better predictions
- Experiment with other models (Logistic Regression, Random Forest)
- Develop interactive dashboards for clinical teams

## References
- GeeksforGeeks. (n.d.). XGBoost Algorithm in Machine Learning. Retrieved from https://www.geeksforgeeks.org/machine-learning/xgboost/
- DRIAMS dataset. Retrieved from https://www.kaggle.com/datasets/drscarlat/driams

## Author
Juan Tavira  
Data Science Master's Graduate with experience healthcare technology

## How to Use
1. Clone this repo
2. Install requirements
3. Run `data_cleaning.py` on raw CSVs
4. Run `classification_model.py` to train and evaluate
