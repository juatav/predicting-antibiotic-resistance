import pandas as pd

# Load the combined strat clean file
df = pd.read_csv("/Users/juanctavira/Desktop/AMR/predicting-antibiotic-resistance/combined_strat_clean.csv")

# Drop any unnamed index columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Normalize all string columns (just in case)
df = df.apply(lambda x: x.str.lower().str.strip() if x.dtypes == "object" else x)

# Define columns to keep as identifiers
id_vars = ['code', 'species', 'year', 'site', 'workstation']

# All other columns are antibiotic results
value_vars = [col for col in df.columns if col not in id_vars and col != 'laboratory_species']

# Reshape to long format
df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                  var_name='antibiotic', value_name='phenotype')

# Keep only valid phenotypes
df_long['phenotype'] = df_long['phenotype'].str.strip()
df_long = df_long[df_long['phenotype'].isin(['s', 'r'])]

# Final check
print(f"Final shape: {df_long.shape}")
print(" Preview:\n", df_long.head())

# Save to file
df_long.to_csv("resistance_long_format.csv", index=False)
print("Saved long-format resistance data to: resistance_long_format.csv")
