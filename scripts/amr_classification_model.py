import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Load and Prepare Dataset
df = pd.read_csv("/Users/juanctavira/Desktop/AMR/predicting-antibiotic-resistance/data/processed/resistance_long_format.csv")

# Keep only 's' and 'r' phenotypes
df = df[df['phenotype'].isin(['s', 'r'])].copy()

# Encode target: 0 = susceptible, 1 = resistant
df['target'] = df['phenotype'].map({'s': 0, 'r': 1})

# Select features and target
features = ['species', 'antibiotic', 'workstation', 'site']
X = df[features]
y = df['target']

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build Pipeline with XGBoost
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), features)],
    remainder='drop'
)

pipeline = Pipeline(steps=[
    ('encoder', encoder),
    ('xgb', xgb.XGBClassifier(eval_metric='logloss'))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# 4. Evaluation Metrics
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 5. Feature Importance Plot
os.makedirs("plots", exist_ok=True)
model = pipeline.named_steps['xgb']
ohe_feature_names = pipeline.named_steps['encoder'].named_transformers_['cat'].get_feature_names_out(features)
importance_scores = model.feature_importances_

# Plot Top 20 Important Features
sorted_idx = np.argsort(importance_scores)[::-1][:20]
plt.figure(figsize=(10, 6))
plt.barh(np.array(ohe_feature_names)[sorted_idx][::-1], importance_scores[sorted_idx][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features in Predicting Resistance")
plt.tight_layout()
plt.savefig("plots/top20_features_in_predicting_resitancerate.png")
plt.show()

# 6. Resistance Rate by Species / Source / Drug
# Resistance rate by species
species_resistance = df.groupby('species')['target'].mean().sort_values(ascending=False)

# Resistance rate by workstation (specimen source)
workstation_resistance = df.groupby('workstation')['target'].mean().sort_values(ascending=False)

# Resistance rate by antibiotic
abx_resistance = df.groupby('antibiotic')['target'].mean().sort_values(ascending=False)

# Barplot: Top 10 Antibiotics by Resistance Rate
abx_resistance.head(10).plot(kind='barh')
plt.title("Top 10 Antibiotics by Resistance Rate")
plt.xlabel("Resistance Rate")
plt.tight_layout()
plt.savefig("plots/top10_antibiotics_by_resitancerate.png")
plt.show()

# Barplot: Top 10 Species by Resistance Rate
top_species = species_resistance.head(10)
sns.barplot(x=top_species.values, y=top_species.index, palette='viridis')
plt.title("Top 10 Species by Resistance Rate")
plt.xlabel("Resistance Rate")
plt.tight_layout()
plt.savefig("plots/top10_species_by_resitancerate.png")
plt.show()

# 7. Heatmap: Species x Abx
# Filter top species and antibiotics
top_species = df['species'].value_counts().head(10).index
top_abx = df['antibiotic'].value_counts().head(10).index
heat_df = df[df['species'].isin(top_species) & df['antibiotic'].isin(top_abx)]

# Pivot to compute mean resistance rate
pivot_table = heat_df.pivot_table(
    index='species',
    columns='antibiotic',
    values='target',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': 'Resistance Rate'})
plt.title("Resistance Rate by Species and Antibiotic")
plt.ylabel("Species")
plt.xlabel("Antibiotic")
plt.tight_layout()
plt.savefig("plots/resistance_heatmap.png")
plt.show()

# 8. Stratified Bar Chart: Resistance by Specimen Source (Workstation) for Top 5 Species
# Top 5 species by frequency
top_species_list = df['species'].value_counts().head(5).index

# Filter data to those species
df_top_species = df[df['species'].isin(top_species_list)]

# Group by species and workstation to get mean resistance (target)
pivot_species_workstation = df_top_species.groupby(['species', 'workstation'])['target'].mean().unstack()

# Plot
pivot_species_workstation.T.plot(kind='bar', figsize=(10, 6))
plt.title("Resistance Rate by Specimen Source for Top 5 Species")
plt.ylabel("Resistance Rate")
plt.xlabel("Workstation (Specimen Source)")
plt.legend(title='Species')
plt.tight_layout()
plt.savefig("plots/resistance_by_species_and_workstation.png")
plt.show()

# 9. Model Performance Breakdown: Precision, Recall, F1-score by Class
# Get the report as a dict
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame for easy plotting
report_df = pd.DataFrame(report_dict).transpose().round(2)

# Only keep the 0 and 1 class rows (susceptible/resistant)
report_df = report_df.loc[['0', '1']]

# Rename rows
report_df.index = ['Susceptible (0)', 'Resistant (1)']

# Plot
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title("Model Performance by Class")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/classification_metrics_by_class.png")
plt.show()

