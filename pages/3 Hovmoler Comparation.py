import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hovmoler MAE Analisis",
    page_icon="🗺️",
    layout="wide"
)

os.environ["STREAMLIT_WATCHDOG"] = "false"

# URL dasar untuk data prediksi
base_url_pred = "data/10k_epoch/pred"
base_url_padanan = "data/10k_epoch/padanan"

# Info dataset yang akan dibandingkan
dataset_info = {
    "0 Variabel": {"folder": "0_var", "prefix": "all_data_0var"},
    "10 Variabel": {"folder": "10_var", "prefix": "all_data_10var"},
    "51 Variabel": {"folder": "51_var", "prefix": "all_data_51var"},
}

# Data stasiun yang baru
station_data = [
    {"name": "Stasiun 218", "lat": -8, "lon": 113.5, "index": 218, "short_name": "S.218"},
    {"name": "Stasiun 294", "lat": -7.5, "lon": 110, "index": 294, "short_name": "S.294"},
    {"name": "Stasiun 329", "lat": -7.25, "lon": 107.5, "index": 329, "short_name": "S.329"},
    {"name": "Stasiun 333", "lat": -7.25, "lon": 108.5, "index": 333, "short_name": "S.333"},
    {"name": "Stasiun 384", "lat": -7, "lon": 110, "index": 384, "short_name": "S.384"},
    {"name": "Stasiun 393", "lat": -7, "lon": 112.25, "index": 393, "short_name": "S.393"},
    {"name": "Stasiun 505", "lat": -6.25, "lon": 106.5, "index": 505, "short_name": "S.505"},
]

@st.cache_data
def load_data(dataset_name: str, tahun: int):
    """Memuat data prediksi dari file Parquet."""
    folder = dataset_info[dataset_name]["folder"]
    prefix = dataset_info[dataset_name]["prefix"]
    url = f"{base_url_pred}/{folder}/{prefix}_{tahun}.parquet"
    try:
        df = pd.read_parquet(url, engine="pyarrow")
        df = df.convert_dtypes()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)
        return df
    except Exception as e:
        st.error(f"❌ Gagal baca file: {url}\nError: {e}")
        return pd.DataFrame()

@st.cache_data
def load_padanan_data(tahun: int):
    """Memuat data curah hujan aktual dari file Parquet."""
    url = f"{base_url_padanan}/CLEANED_PADANAN_{tahun}.parquet"
    try:
        df = pd.read_parquet(url, engine="pyarrow")
        if 'lon' in df.columns: df = df.rename(columns={'lon': 'longitude'})
        if 'lat' in df.columns: df = df.rename(columns={'lat': 'latitude'})
        if 'idx_new' in df.columns: df = df.rename(columns={'idx_new': 'idx'})
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca file padanan: {url}\nError: {e}")
        return pd.DataFrame()
    
    required_cols = ['month', 'year', 'latitude', 'longitude', 'rainfall']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    return df[required_cols]

def calculate_mae_by_year_and_station(data_dict, years, selected_stations_list):
    """Menghitung MAE per tahun untuk stasiun-stasiun yang dipilih."""
    mae_list = []
    all_stations = {s['name']: (s['lat'], s['lon']) for s in station_data}
    
    # Filter stasiun berdasarkan input multiselect
    stations_to_process = selected_stations_list
    
    for model_name, df_full in data_dict.items():
        if df_full.empty: continue
        for station_name in stations_to_process:
            lat, lon = all_stations[station_name]
            df_station = df_full[
                (df_full['latitude'] == lat) & 
                (df_full['longitude'] == lon)
            ].copy()
            if df_station.empty: continue
            df_mae = df_station.groupby('year').apply(
                lambda x: abs(x['ch_pred'] - x['rainfall']).mean()
            ).reset_index()
            df_mae.columns = ['year', 'mae']
            df_mae['model'] = model_name
            df_mae['station'] = station_name
            mae_list.append(df_mae)
    if mae_list:
        combined_mae_df = pd.concat(mae_list, ignore_index=True)
        return combined_mae_df
    else:
        return pd.DataFrame()

# --- Main Streamlit app logic ---
st.title("🗺️ DK Viewer - Diagram Hovmoler MAE")
st.markdown("""
Visualisasi ini menunjukkan performa MAE dari tiga model (0, 10, dan 51 variabel) per tahun untuk setiap stasiun. 
Pilih stasiun mana pun untuk membuat diagram Hovmoler.
""")

### Blok Konfigurasi di Sidebar
with st.sidebar.form("config_form"):
    st.header("⚙️ Konfigurasi")
    plot_choice = st.radio("Pilih Pustaka Visualisasi:", ('Plotly Pixel', 'Matplotlib Contour'))
    year_options = list(range(1985, 2015))
    
    col1, col2 = st.columns(2)
    with col1:
        tahun_from = st.selectbox("Tahun Awal:", year_options, index=0)
    with col2:
        tahun_until = st.selectbox("Tahun Akhir:", year_options, index=len(year_options) - 1)
    
    # Opsi pengaturan batas skala warna
    st.subheader("Batas Skala Warna MAE (Z-Axis)")
    scale_mode = st.radio(
        "Mode Skala Warna:",
        ('Otomatis (Data Min/Max)', 'Kustom'),
        key='scale_mode'
    )

    custom_zmin = 0.0
    custom_zmax = 800.0 # DIUBAH: Default Zmax menjadi 800

    if scale_mode == 'Kustom':
        col3, col4 = st.columns(2)
        with col3:
            # Pilihan kustom Z-min (Default: 0)
            custom_zmin = st.number_input(
                "MAE Minimum (Zmin):", 
                min_value=0.0, 
                value=0.0, 
                step=1.0,
                format="%f",
                key='zmin_input'
            )
        with col4:
            # Pilihan kustom Z-max (Default: 800)
            custom_zmax = st.number_input(
                "MAE Maksimum (Zmax):", 
                min_value=1.0, 
                value=800.0, # DIUBAH: Default Zmax menjadi 800
                step=10.0,
                format="%f",
                key='zmax_input'
            )
    
    # Ganti st.selectbox dengan st.multiselect
    station_names = [s['name'] for s in station_data]
    selected_stations = st.multiselect(
        "Pilih Stasiun:", 
        options=station_names, 
        default=station_names
    )

    submit = st.form_submit_button("🚀 Buat Diagram Hovmoler")

### Blok Utama yang Hanya Tampil Setelah Tombol Submit Ditekan
if submit:
    if tahun_from > tahun_until:
        st.error("❌ Tahun 'Awal' tidak boleh lebih baru dari tahun 'Akhir'.")
    elif scale_mode == 'Kustom' and custom_zmin >= custom_zmax:
        st.error("❌ Zmin harus lebih kecil dari Zmax saat menggunakan mode Kustom.")
    elif not selected_stations:
        st.warning("⚠️ Silakan pilih setidaknya satu stasiun untuk menampilkan diagram.")
    else:
        years_to_process = list(range(tahun_from, tahun_until + 1))
        
        with st.spinner("⏳ Memuat dan memproses data... Ini mungkin butuh waktu beberapa menit."):
            all_data_by_model = {}
            for model_name in dataset_info.keys():
                all_filtered_years = []
                for year in years_to_process:
                    df_pred = load_data(model_name, year)
                    df_actual = load_padanan_data(year)
                    if not df_pred.empty and not df_actual.empty:
                        df_merged = pd.merge(df_pred, df_actual, on=['month', 'year', 'latitude', 'longitude'], how='left')
                        all_filtered_years.append(df_merged)
                if all_filtered_years:
                    all_data_by_model[model_name] = pd.concat(all_filtered_years, ignore_index=True)
                else:
                    all_data_by_model[model_name] = pd.DataFrame()
            
            # Panggil fungsi dengan list stasiun yang dipilih
            mae_results = calculate_mae_by_year_and_station(all_data_by_model, years_to_process, selected_stations)
        
        if mae_results.empty:
            st.warning("⚠️ Tidak ada data MAE yang dapat dihitung untuk stasiun yang dipilih. Pastikan file data tersedia.")
        else:
            st.success("✅ Data berhasil diproses. Diagram Hovmoler siap.")
            
            # --- TENTUKAN BATAS ZMIN/ZMAX BERDASARKAN MODE YANG DIPILIH ---
            if scale_mode == 'Otomatis (Data Min/Max)':
                # Batas otomatis: 0 dan nilai maksimum data MAE yang ada
                final_zmin = 0.0 # MAE tidak mungkin negatif
                final_zmax = mae_results['mae'].max() * 1.05 # Tambahkan sedikit buffer (5%)
            else:
                # Batas kustom
                final_zmin = custom_zmin
                final_zmax = custom_zmax
            
            # --- Urutan stasiun yang dipilih (disesuaikan dengan urutan multiselect)
            station_order_short_dict = {s['name']: s['short_name'] for s in station_data}
            station_order_short = [station_order_short_dict[name] for name in selected_stations]
            station_names_ordered = selected_stations

            # --- Pilih plot berdasarkan opsi pengguna ---
            if plot_choice == 'Plotly Pixel':
                st.subheader("Diagram Hovmoler (Plotly)")
                
                fig = make_subplots(
                    rows=1, cols=len(dataset_info),
                    subplot_titles=list(dataset_info.keys()),
                    shared_yaxes=True, horizontal_spacing=0.05
                )
                for i, model_name in enumerate(dataset_info.keys()):
                    df_plot = mae_results[mae_results['model'] == model_name].copy()
                    years_df = pd.DataFrame({'year': years_to_process})
                    stations_df = pd.DataFrame({'station': station_names_ordered})
                    full_grid = pd.MultiIndex.from_product([years_df['year'], stations_df['station']], names=['year', 'station'])
                    df_plot = df_plot.set_index(['year', 'station']).reindex(full_grid).reset_index()
                    df_pivot = df_plot.pivot(index='year', columns='station', values='mae')
                    df_pivot = df_pivot.reindex(columns=station_names_ordered)
                    station_short_names = {s['name']: s['short_name'] for s in station_data}
                    hover_text = df_pivot.apply(lambda col: [
                        f"Stasiun: {station_short_names[col.name]}<br>Tahun: {year}<br>MAE: {value:.2f}"
                        for year, value in col.items()
                    ], axis=0).values.tolist()
                    heatmap = go.Heatmap(
                        z=df_pivot.values, x=[station_short_names[name] for name in df_pivot.columns],
                        y=df_pivot.index, colorscale=[[0, 'rgb(240, 248, 255)'], [1, 'rgb(0, 0, 128)']],
                        zmin=final_zmin, zmax=final_zmax, # Menggunakan nilai final_zmin/zmax
                        colorbar=dict(title='MAE', thickness=20) if i == 0 else None,
                        showscale=i == 0, hoverinfo='text', text=hover_text,
                    )
                    fig.add_trace(heatmap, row=1, col=i + 1)
                fig.update_layout(title_text="Diagram Hovmoler MAE per Tahun dan Lokasi Stasiun", height=800, width=1200)
                fig.update_yaxes(title_text="Tahun", tickmode='linear', dtick=2, row=1, col=1)
                fig.update_xaxes(title_text="Stasiun", side="bottom", tickvals=station_order_short, ticktext=station_order_short)
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_choice == 'Matplotlib Contour':
                st.subheader("Diagram Hovmoler (Matplotlib)")
                
                fig, axes = plt.subplots(1, len(dataset_info), figsize=(20, 10), sharey=True)
                
                for i, model_name in enumerate(dataset_info.keys()):
                    df_plot = mae_results[mae_results['model'] == model_name].copy()
                    years_df = pd.DataFrame({'year': years_to_process})
                    stations_df = pd.DataFrame({'station': station_names_ordered})
                    full_grid = pd.MultiIndex.from_product([years_df['year'], stations_df['station']], names=['year', 'station'])
                    df_plot = df_plot.set_index(['year', 'station']).reindex(full_grid).reset_index()
                    df_pivot = df_plot.pivot(index='year', columns='station', values='mae')
                    df_pivot = df_pivot.reindex(columns=station_names_ordered)
                    ax = axes[i]
                    im = ax.contourf(
                        df_pivot.columns, df_pivot.index, df_pivot.values,
                        # levels disesuaikan dengan nilai final_zmin/zmax
                        levels=np.linspace(final_zmin, final_zmax, 20), cmap='viridis', extend='both'
                    )
                    ax.set_title(model_name)
                    ax.set_xlabel("Stasiun")
                    ax.set_xticks(range(len(station_order_short)))
                    ax.set_xticklabels(station_order_short, rotation=45, ha='right')
                    ax.set_yticks(df_pivot.index[::2])
                    ax.set_ylim(tahun_from, tahun_until)
                    ax.set_aspect('auto')
                    if i == 0:
                        ax.set_ylabel("Tahun")
                
                fig.suptitle("Diagram Hovmoler MAE per Tahun dan Lokasi Stasiun", y=1.02, fontsize=16)
                fig.tight_layout()
                fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.7, label='MAE')
                st.pyplot(fig)

            # --- Peta Lokasi Stasiun (Tampil di kedua opsi) ---
            st.header("📍 Peta Lokasi Stasiun")
            st.markdown("Peta ini menunjukkan lokasi geografis stasiun yang dianalisis. Anda bisa memperbesar untuk melihat detailnya.")
            df_stations = pd.DataFrame(station_data)
            fig_map = px.scatter_mapbox(
                df_stations, lat="lat", lon="lon", hover_name="name",
                hover_data={'name': False, 'lat': True, 'lon': True},
                color_discrete_sequence=["#0068c9"], size=[12] * len(df_stations),
                zoom=7, height=300, title="Lokasi Stasiun Pengamatan"
            )
            for _, row in df_stations.iterrows():
                fig_map.add_trace(go.Scattermapbox(
                    lat=[row['lat']], lon=[row['lon']], mode='text',
                    text=[row['short_name']], textposition="bottom right",
                    textfont=dict(size=10, color="black"), showlegend=False
                ))
            fig_map.update_layout(
                mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0},
                mapbox=dict(center=go.layout.mapbox.Center(lat=-7.5, lon=110), zoom=7),
                showlegend=False
            )
            st.plotly_chart(fig_map, use_container_width=False)
            st.markdown("---")

            st.subheader("Data MAE Detail")
            st.dataframe(mae_results)
            
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