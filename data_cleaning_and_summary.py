import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load merged data
merged = pd.read_csv('output/merged_data.csv')

# Convert 'Date' to datetime
merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')

# Create a 'CompetitionOpenDate' column from year and month
# If either is missing, set as NaT
merged['CompetitionOpenSinceYear'] = merged['CompetitionOpenSinceYear'].astype('Int64')
merged['CompetitionOpenSinceMonth'] = merged['CompetitionOpenSinceMonth'].astype('Int64')
merged['CompetitionOpenDate'] = pd.to_datetime(
    merged['CompetitionOpenSinceYear'].astype(str) + '-' + merged['CompetitionOpenSinceMonth'].astype(str) + '-01',
    errors='coerce'
)

# Save the cleaned dataframe
merged.to_csv('output/merged_data_dates_converted.csv', index=False)

# Describe numerical columns
summary = merged.describe()
summary.to_csv('output/merged_data_summary.csv')
print(summary)
print('Date columns converted and summary saved to output/merged_data_summary.csv')

# Ensure output directory exists
os.makedirs('output/eda_plots', exist_ok=True)

# Time-series plot of Sales over Date
if 'Sales' in merged.columns and 'Date' in merged.columns:
    plt.figure(figsize=(14,6))
    merged.groupby('Date')['Sales'].sum().plot()
    plt.title('Total Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('output/eda_plots/sales_over_time.png')
    plt.close()

# Boxplots: Sales vs Promo, DayOfWeek, SchoolHoliday
for col in ['Promo', 'DayOfWeek', 'SchoolHoliday']:
    if col in merged.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x=merged[col], y=merged['Sales'])
        plt.title(f'Sales by {col}')
        plt.xlabel(col)
        plt.ylabel('Sales')
        plt.tight_layout()
        plt.savefig(f'output/eda_plots/sales_by_{col.lower()}.png')
        plt.close()

# Histogram of CompetitionDistance
if 'CompetitionDistance' in merged.columns:
    plt.figure(figsize=(8,6))
    sns.histplot(merged['CompetitionDistance'].dropna(), bins=50, kde=True)
    plt.title('Distribution of CompetitionDistance')
    plt.xlabel('CompetitionDistance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('output/eda_plots/competition_distance_hist.png')
    plt.close() 