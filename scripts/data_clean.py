import os
import pandas as pd

# Base directory where DRIAMS data is located
base_dir = "/Users/juanctavira/Desktop/AMR/predicting-antibiotic-resistance/driams_dataset"
site = "A"
years = ["2015", "2016", "2017", "2018"]

# Collect data
all_years = []

for year in years:
    file_path = os.path.join(base_dir, f"DRIAMS-{site}", "id", year, f"{year}_strat.csv")

    try:
        print(f"\n Loading {year}...")
        df = pd.read_csv(file_path, low_memory=False)

        # Strip column names
        df.columns = df.columns.str.strip()

        # Fix known issues — only drop rows with missing critical fields
        df = df.dropna(subset=["code", "species"])

        # Clean string fields only (don't drop entire rows unnecessarily)
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # Add metadata columns
        df["year"] = year
        df["site"] = site

        # Output debug info
        print(f" Rows retained for {year}: {len(df)}")
        print(f" Columns: {df.columns.tolist()}")

        # Save per-year cleaned file
        out_path = os.path.join(base_dir, f"DRIAMS-{site}", "id", year, f"{year}_strat_clean.csv")
        df.to_csv(out_path, index=False)
        print(f" Saved to {out_path}")

        all_years.append(df)

    except FileNotFoundError:
        print(f" File not found: {file_path}")
    except Exception as e:
        print(f" Error processing {year}: {e}")

# Combine all years
if all_years:
    combined = pd.concat(all_years, ignore_index=True)
    combined_out = os.path.join(base_dir, f"DRIAMS-{site}", "combined_strat_clean.csv")
    combined.to_csv(combined_out, index=False)
    print(f"\n Combined dataset saved to: {combined_out}")
else:
    print(" No data combined. Check source files.")
