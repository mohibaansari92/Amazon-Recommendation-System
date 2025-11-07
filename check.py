import pandas as pd
df = pd.read_csv('MultipleFiles/Reviews.csv')
print("Exact Column Names:")
print(df.columns.tolist())