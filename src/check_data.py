import os
import pandas as pd

DATA_DIR = "data/raw/complaintsfull"

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("cannot find the data please check the file path data/raw/complaintsfull")

file_path = os.path.join(DATA_DIR, csv_files[0])

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()

print("file path：", file_path)
print("\n column names：")
print(df.columns.tolist())

print("\n first 3 ：")
print(df.head(3).T)

print("\n total：", len(df))