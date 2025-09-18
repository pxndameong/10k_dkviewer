# pages/Data Viewer.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import os

st.set_page_config(
    page_title="Peta Curah Hujan",
    page_icon="🗺️",
    layout="wide"  # Tambahkan baris ini
)

os.environ["STREAMLIT_WATCHDOG"] = "false"

# Pastikan folder data Anda berada di lokasi yang benar
base_url = "data/100k_epoch/pred"

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

# Fungsi untuk membuat peta menggunakan Plotly Express (tidak diubah)
def create_plotly_map(df, point_size, title):
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="ch_pred",
        color_continuous_scale=px.colors.sequential.Rainbow,
        zoom=5,
        mapbox_style="open-street-map",
        hover_data={"year": True, "month": True, "ch_pred": ':.2f'}
    )
    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Curah Hujan (mm)"),
        title_text=title,
        title_x=0.5
    )
    return fig

# Fungsi untuk membuat peta menggunakan Matplotlib Basemap dengan grid penuh
# Fungsi untuk membuat peta menggunakan Matplotlib Basemap dengan resolusi 0.25 derajat
def create_matplotlib_map(df, point_size, title, vmax_ch):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ambil nilai min/max dari data dan tambahkan padding
    lat_min, lat_max = df['latitude'].min() - 0.5, df['latitude'].max() + 0.5
    lon_min, lon_max = df['longitude'].min() - 0.5, df['longitude'].max() + 0.5
    
    # Inisialisasi peta dengan batas yang dinamis
    m = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,
                llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='i', ax=ax)

    # BARIS PERUBAHAN UTAMA:
    # 1. Buat grid bujur dan lintang dengan resolusi 0.25 derajat
    lon_grid, lat_grid = np.meshgrid(
        np.arange(lon_min, lon_max + 0.25, 0.25),  # UBAH STEP DARI 1 KE 0.25
        np.arange(lat_min, lat_max + 0.25, 0.25)   # UBAH STEP DARI 1 KE 0.25
    )
    
    # 2. Lakukan interpolasi data curah hujan ke grid
    points = df[['longitude', 'latitude']].values
    values = df['ch_pred'].values
    ch_grid = griddata(points, values, (lon_grid, lat_grid), method='cubic')
    
    # 3. Gunakan m.pcolormesh untuk memplot grid penuh
    x, y = m(lon_grid, lat_grid)
    sc = m.pcolormesh(x, y, ch_grid, cmap='rainbow', vmin=0, vmax=vmax_ch)

    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    
    # Tambahkan grid dan label lintang/bujur (lat/lon)
    # Atur selisih grid ke 1 derajat agar tidak terlalu padat
    m.drawparallels(np.arange(int(lat_min)-1, int(lat_max)+2, 1), labels=[1,0,0,0], fmt='%.1f')
    m.drawmeridians(np.arange(int(lon_min)-1, int(lon_max)+2, 1), labels=[0,0,0,1], fmt='%.1f')
    
    # Membuat colorbar yang lebih panjang dan seimbang menggunakan make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Curah Hujan (mm)')

    ax.set_title(title)
    
    plt.tight_layout()
    
    return fig

# Main Streamlit app logic
st.title("📊 DK Viewer - Data Viewer")

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
    row_options = [i for i in range(50, 1001, 50)]
    max_rows = st.selectbox("Maksimal baris ditampilkan:", row_options, index=0)
    point_size = st.slider("Ukuran titik di peta:", 1, 20, 8, step=1)
    
    map_library_choice = st.radio(
        "Pilih versi peta:",
        ("Plotly", "Matplotlib")
    )
    
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
        if 'global_max_ch' in st.session_state:
            del st.session_state.global_max_ch
    else:
        df_filtered_all = pd.concat(all_filtered, ignore_index=True)
        df_filtered_all['bulan_tahun'] = df_filtered_all['month'].map(bulan_dict) + ' ' + df_filtered_all['year'].astype(str)
        unique_combinations = df_filtered_all['bulan_tahun'].unique()
        
        sorted_combinations = sorted(unique_combinations, key=lambda x: (int(x.rsplit(' ', 1)[1]), [k for k, v in bulan_dict.items() if v == x.rsplit(' ', 1)[0]][0]))
        
        st.session_state.data = df_filtered_all
        st.session_state.combinations = sorted_combinations
        # --- TAMBAHKAN BARIS INI: Simpan nilai maksimum ke session state ---
        st.session_state.global_max_ch = df_filtered_all['ch_pred'].max()
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

    if df_display.empty:
        st.warning("⚠️ Data tidak ditemukan untuk kombinasi yang dipilih.")
    else:
        st.success(f"✅ Menampilkan data untuk {selected_combo} ({len(df_display)} baris)")
        st.write(f"### Preview Data ({max_rows} baris)")
        st.dataframe(df_display.head(max_rows))

        st.write(f"### Peta Curah Hujan {selected_combo}")
                
        if map_library_choice == "Plotly":
                fig = create_plotly_map(df_display, point_size, f"DK Rainfall_{selected_combo}")
                st.plotly_chart(fig, use_container_width=True)
        elif map_library_choice == "Matplotlib":
                # --- TAMBAHKAN BLOK INI: Meneruskan nilai maksimum ---
            if 'global_max_ch' in st.session_state:
                fig = create_matplotlib_map(df_display, point_size, f"DK Rainfall_{selected_combo}", st.session_state.global_max_ch)
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("⚠️ Nilai maksimum curah hujan tidak ditemukan. Silakan submit konfigurasi ulang.")
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .fade-in-text {
        animation: fadeIn 2s ease-in-out;
        text-align: center;
        margin-top: 20px;
    }
    </style>

    <div class="fade-in-text">
        <h4>BRIN Research Team</h4>
        <p><em>Data Visualization by Tsaqib</em></p>
    </div>
    """,
    unsafe_allow_html=True
)