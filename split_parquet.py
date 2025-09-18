import pandas as pd
import os

# Folder input/output
input_file_0var = "data/100k_epoch/parquet_files/all_data_0var_100k_1985-2014.parquet"
input_file_10var = "data/100k_epoch/parquet_files/all_data_10var_100k_1985-2014.parquet"
input_file_51var = "data/100k_epoch/parquet_files/all_data_51var_100k_1985-2014.parquet"

output_folder = "data/100k_epoch/pred/split_by_year"

# Pastikan folder output ada
for sub in ["0_var", "10_var", "51_var"]:
    os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

def split_parquet_by_year(input_file, var_folder, prefix):
    print(f"📥 Membaca {input_file} ...")
    df = pd.read_parquet(input_file, engine="pyarrow")

    if "year" not in df.columns:
        raise ValueError(f"❌ Kolom 'year' tidak ditemukan di {input_file}")

    for year in sorted(df["year"].unique()):
        df_year = df[df["year"] == year]
        out_path = os.path.join(output_folder, var_folder, f"{prefix}_{year}.parquet")
        df_year.to_parquet(out_path, engine="pyarrow", index=False)
        print(f"✅ Simpan {out_path} ({len(df_year)} baris)")

# Jalankan untuk 3 file utama
split_parquet_by_year(input_file_0var, "0_var", "all_data_0var")
split_parquet_by_year(input_file_10var, "10_var", "all_data_10var")
split_parquet_by_year(input_file_51var, "51_var", "all_data_51var")
