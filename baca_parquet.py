import pandas as pd

# Path file parquet
file_path = r"C:\Users\HP Elitebook X360\vscode\TSAQIB\pxnda.venv\Kode Pxnda\dk_viewer\data\100k_epoch\pred\split_by_year\0_var\all_data_0var_1985.parquet"

# Baca parquet
df = pd.read_parquet(file_path)

# Tampilkan 50 baris pertama
print(df.head(200))

# Kalau ingin tahu jumlah baris & kolom
print("\nShape:", df.shape)
