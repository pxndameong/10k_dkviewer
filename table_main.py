import streamlit as st
import pandas as pd
import plotly.express as px
import os
os.environ["STREAMLIT_WATCHDOG"] = "false"

base_url = "https://raw.githubusercontent.com/pxndameong/dkviewer/main/data/10k_epoch/pred"

dataset_info = {
    "0 Variabel": {"folder": "0_var", "prefix": "all_data_0var"},
    "10 Variabel": {"folder": "10_var", "prefix": "all_data_10var"},
    "51 Variabel": {"folder": "51_var", "prefix": "all_data_51var"},
}

bulan_dict = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Data stasiun yang baru
station_data = [
    {"name": "Stasiun 218 (Lat: -8, Lon: 113.5)", "lat": -8, "lon": 113.5, "index": 218},
    {"name": "Stasiun 294 (Lat: -7.5, Lon: 110)", "lat": -7.5, "lon": 110, "index": 294},
    {"name": "Stasiun 329 (Lat: -7.25, Lon: 107.5)", "lat": -7.25, "lon": 107.5, "index": 329},
    {"name": "Stasiun 333 (Lat: -7.25, Lon: 108.5)", "lat": -7.25, "lon": 108.5, "index": 333},
    {"name": "Stasiun 384 (Lat: -7, Lon: 110)", "lat": -7, "lon": 110, "index": 384},
    {"name": "Stasiun 393 (Lat: -7, Lon: 112.25)", "lat": -7, "lon": 112.25, "index": 393},
    {"name": "Stasiun 505 (Lat: -6.25, Lon: 106.5)", "lat": -6.25, "lon": 106.5, "index": 505},
]
station_names = ["Semua Stasiun"] + [s["name"] for s in station_data]

@st.cache_data
def load_data(dataset_name: str, tahun: int):
    folder = dataset_info[dataset_name]["folder"]
    prefix = dataset_info[dataset_name]["prefix"]
    url = f"{base_url}/{folder}/{prefix}_{tahun}.parquet"
    try:
        df = pd.read_parquet(url, engine="pyarrow")
    except Exception as e:
        st.error(f"❌ Gagal baca file: {url}\nError: {e}")
        return pd.DataFrame()
    df = df.convert_dtypes()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def main():
    st.title("📊 DK Viewer - 10K Epoch")

    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'combinations' not in st.session_state:
        st.session_state.combinations = []

    with st.sidebar.form("config_form"):
        st.header("⚙️ Konfigurasi")
        dataset_choice = st.selectbox("Pilih dataset:", list(dataset_info.keys()))
        year_options = ["All"] + list(range(1985, 2015))
        tahun_pilih = st.multiselect("Pilih tahun:", year_options, default=["All"])
        bulan_options = ["All"] + list(range(1, 13))
        bulan_pilih = st.multiselect(
            "Pilih bulan:",
            bulan_options,
            default=["All"],
            format_func=lambda x: bulan_dict[x] if x != "All" else "All"
        )
        # Dropdown baru untuk stasiun
        selected_station_name = st.selectbox("Pilih stasiun:", station_names)
        
        row_options = [i for i in range(50, 1001, 50)]
        max_rows = st.selectbox("Maksimal baris ditampilkan:", row_options, index=0)
        submit = st.form_submit_button("🚀 Submit konfigurasi")

    if submit:
        if "All" in tahun_pilih:
            tahun_final = list(range(1985, 2015))
        else:
            tahun_final = [t for t in tahun_pilih if t != "All"]
        if "All" in bulan_pilih:
            bulan_final = list(range(1, 13))
        else:
            bulan_final = [b for b in bulan_pilih if b != "All"]

        all_filtered = []
        for th in tahun_final:
            df = load_data(dataset_choice, th)
            if not df.empty:
                df = df[df["month"].isin(bulan_final)]
                all_filtered.append(df)
        
        if not all_filtered:
            st.warning("⚠️ Tidak ada data sesuai filter.")
            st.session_state.data = None
            st.session_state.combinations = []
        else:
            df_filtered_all = pd.concat(all_filtered, ignore_index=True)
            df_filtered_all['bulan_tahun'] = df_filtered_all['month'].map(bulan_dict) + ' ' + df_filtered_all['year'].astype(str)
            unique_combinations = df_filtered_all['bulan_tahun'].unique()
            
            # Sortir kombinasi berdasarkan tahun dan bulan
            sorted_combinations = sorted(unique_combinations, key=lambda x: (int(x.rsplit(' ', 1)[1]), [k for k, v in bulan_dict.items() if v == x.rsplit(' ', 1)[0]][0]))
            
            st.session_state.data = df_filtered_all
            st.session_state.combinations = sorted_combinations
            st.success("✅ Data berhasil dimuat. Silakan pilih dari menu dropdown.")
    
    if st.session_state.data is not None and len(st.session_state.combinations) > 0:
        st.subheader("Pilih Bulan dan Tahun untuk Dilihat")
        selected_combo = st.selectbox("Pilih data yang ingin ditampilkan:", st.session_state.combinations)
        
        selected_month_str, selected_year_str = selected_combo.rsplit(' ', 1)
        selected_month_num = [key for key, val in bulan_dict.items() if val == selected_month_str][0]
        selected_year_num = int(selected_year_str)

        df_display = st.session_state.data[
            (st.session_state.data['month'] == selected_month_num) & 
            (st.session_state.data['year'] == selected_year_num)
        ]

        # Logika filter stasiun
        if selected_station_name != "Semua Stasiun":
            station_info = next(s for s in station_data if s["name"] == selected_station_name)
            df_display = df_display[
                (df_display['latitude'] == station_info['lat']) & 
                (df_display['longitude'] == station_info['lon'])
            ]
        
        if df_display.empty:
            st.warning("⚠️ Data tidak ditemukan untuk kombinasi yang dipilih.")
            return

        st.success(f"✅ Menampilkan data untuk {selected_combo} ({len(df_display)} baris)")
        st.write(f"### Preview Data ({max_rows} baris)")
        st.dataframe(df_display.head(max_rows))