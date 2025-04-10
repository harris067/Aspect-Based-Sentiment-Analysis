import pandas as pd


file1 = pd.read_csv('aug_data.csv')
file2 = pd.read_csv('adversarial_data.csv')


max_len = max(len(file1), len(file2))


file1 = file1.reindex(range(max_len))
file2 = file2.reindex(range(max_len))


merged = pd.DataFrame()
for i in range(max_len):
    merged = pd.concat([merged, file1.iloc[[i]], file2.iloc[[i]]], ignore_index=True)


merged.to_csv('merged.csv', index=False)
