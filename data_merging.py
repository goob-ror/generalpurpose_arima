import pandas as pd

# Load datasets
train = pd.read_csv('rossmann-store-sales/train.csv')
test = pd.read_csv('rossmann-store-sales/test.csv')
store = pd.read_csv('rossmann-store-sales/store.csv')

# Merge train and store
merged_train = pd.merge(train, store, on='Store', how='left')
merged_train.to_csv('output/merged_data.csv', index=False)

# Merge test and store
merged_test = pd.merge(test, store, on='Store', how='left')
merged_test.to_csv('output/merged_test_data.csv', index=False)

print('Merging complete. Files saved as merged_data.csv and merged_test_data.csv.') 