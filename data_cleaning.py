import pandas as pd
import numpy as np

# Load merged data
merged = pd.read_csv('output/merged_data.csv')

# Columns to impute
num_impute_cols = [
    'CompetitionDistance',
    'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear',
    'Promo2SinceWeek',
    'Promo2SinceYear'
]
cat_impute_cols = ['PromoInterval']

# Impute numerical columns with median
for col in num_impute_cols:
    if col in merged.columns:
        median = merged[col].median()
        merged[col].fillna(median, inplace=True)

# Impute categorical columns with mode
for col in cat_impute_cols:
    if col in merged.columns:
        mode = merged[col].mode()[0]
        merged[col].fillna(mode, inplace=True)

# Handle outliers in Sales and Customers using IQR
for col in ['Sales', 'Customers']:
    if col in merged.columns:
        Q1 = merged[col].quantile(0.25)
        Q3 = merged[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Cap outliers
        merged[col] = np.where(merged[col] < lower, lower, merged[col])
        merged[col] = np.where(merged[col] > upper, upper, merged[col])

# Treat 1900 in CompetitionOpenSinceYear as missing
if 'CompetitionOpenSinceYear' in merged.columns:
    merged.loc[merged['CompetitionOpenSinceYear'] == 1900, 'CompetitionOpenSinceYear'] = np.nan

# Save cleaned data
merged.to_csv('output/cleaning_data.csv', index=False)
print('Cleaning complete. File saved as cleaning_data.csv.') 