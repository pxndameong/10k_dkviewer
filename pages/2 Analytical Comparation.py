# pages/Analytical Table.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(
    page_title="Tabel Data Komparatif",
    page_icon="📊",
    layout="wide"
)

os.environ["STREAMLIT_WATCHDOG"] = "false"

# URL dasar untuk data prediksi
base_url_pred = "data/100k_epoch/pred"

# URL dasar untuk data padanan
base_url_padanan = "data/100k_epoch/padanan"

# Info dataset yang akan dibandingkan
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
station_names = [s["name"] for s in station_data]

@st.cache_data
def load_data(dataset_name: str, tahun: int):
    folder = dataset_info[dataset_name]["folder"]
    prefix = dataset_info[dataset_name]["prefix"]
    url = f"{base_url_pred}/{folder}/{prefix}_{tahun}.parquet"
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

@st.cache_data
def load_padanan_data(tahun: int):
    """
    Fungsi untuk memuat data padanan.
    """
    url = f"{base_url_padanan}/CLEANED_PADANAN_{tahun}.parquet"
    
    try:
        df = pd.read_parquet(url, engine="pyarrow")
        if 'lon' in df.columns:
            df = df.rename(columns={'lon': 'longitude'})
        if 'lat' in df.columns:
            df = df.rename(columns={'lat': 'latitude'})
        if 'idx_new' in df.columns:
            df = df.rename(columns={'idx_new': 'idx'})
            
    except Exception as e:
        st.warning(f"⚠️ Gagal membaca file padanan: {url}\nError: {e}")
        return pd.DataFrame()
        
    required_cols = ['month', 'year', 'latitude', 'longitude', 'rainfall', 'idx']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    
    return df[required_cols]

# Main Streamlit app logic
st.title("📊 DK Viewer - Tabel Analisis Komparatif")

if 'comparative_data' not in st.session_state:
    st.session_state.comparative_data = None
if 'combinations' not in st.session_state:
    st.session_state.combinations = []

with st.sidebar.form("config_form"):
    st.header("⚙️ Konfigurasi")
    
    year_options = list(range(1985, 2015))
    bulan_options = list(range(1, 13))
    
    st.subheader("Dari")
    col1, col2 = st.columns(2)
    with col1:
        bulan_from = st.selectbox(
            "Bulan Awal:",
            bulan_options,
            index=0,
            format_func=lambda x: bulan_dict[x]
        )
    with col2:
        tahun_from = st.selectbox("Tahun Awal:", year_options, index=0)
    
    st.subheader("Sampai")
    col3, col4 = st.columns(2)
    with col3:
        bulan_until = st.selectbox(
            "Bulan Akhir:",
            bulan_options,
            index=len(bulan_options) - 1,
            format_func=lambda x: bulan_dict[x]
        )
    with col4:
        tahun_until = st.selectbox("Tahun Akhir:", year_options, index=len(year_options) - 1)
        
    selected_station_name = st.selectbox("Pilih stasiun:", station_names)
    
    submit = st.form_submit_button("🚀 Submit konfigurasi dan bandingkan")

if submit:
    from_date = (tahun_from, bulan_from)
    until_date = (tahun_until, bulan_until)

    if from_date > until_date:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih baru dari tanggal 'Sampai'.")
    else:
        tahun_final = list(range(tahun_from, tahun_until + 1))
        
        filtered_data_dict = {}
        
        for dataset_name in dataset_info.keys():
            all_filtered = []
            for th in tahun_final:
                df_main = load_data(dataset_name, th)
                df_padanan = load_padanan_data(th)
                
                if not df_main.empty and not df_padanan.empty:
                    df_merged_year = pd.merge(df_main, df_padanan, on=['month', 'year', 'latitude', 'longitude'], how='left')
                    df_merged_year = df_merged_year.drop_duplicates(subset=['latitude', 'longitude', 'month', 'year'])
                    all_filtered.append(df_merged_year)
                elif not df_main.empty:
                    all_filtered.append(df_main)
            
            if all_filtered:
                df_filtered_all = pd.concat(all_filtered, ignore_index=True)
                
                df_filtered_all['error_bias'] = df_filtered_all['ch_pred'] - df_filtered_all['rainfall']
                df_filtered_all['absolute_error'] = abs(df_filtered_all['ch_pred'] - df_filtered_all['rainfall'])
                df_filtered_all['squared_error'] = (df_filtered_all['ch_pred'] - df_filtered_all['rainfall'])**2

                station_info = next(s for s in station_data if s["name"] == selected_station_name)
                df_filtered_station = df_filtered_all[
                    (df_filtered_all['latitude'] == station_info['lat']) & 
                    (df_filtered_all['longitude'] == station_info['lon'])
                ].copy()

                filtered_data_dict[dataset_name] = df_filtered_station
            else:
                filtered_data_dict[dataset_name] = pd.DataFrame()

        st.session_state.comparative_data = filtered_data_dict
        st.success("✅ Data berhasil dimuat dan siap untuk perbandingan.")

if st.session_state.comparative_data:
    st.markdown("---")
    st.subheader(f"Ringkasan Statistik Komparatif untuk Stasiun {selected_station_name}")
    
    summary_cols = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error']
    comparison_summary = []
    
    for dataset_name, df in st.session_state.comparative_data.items():
        if not df.empty:
            summary_row = {"Metrik": dataset_name}
            for col in summary_cols:
                if col in df.columns:
                    summary_row[f"Mean ({col})"] = df[col].mean()
                    summary_row[f"Sum ({col})"] = df[col].sum()
                else:
                    summary_row[f"Mean ({col})"] = None
                    summary_row[f"Sum ({col})"] = None
            comparison_summary.append(summary_row)
        else:
            summary_row = {"Metrik": dataset_name}
            for col in summary_cols:
                summary_row[f"Mean ({col})"] = None
                summary_row[f"Sum ({col})"] = None
            comparison_summary.append(summary_row)

    if comparison_summary:
        comparison_df = pd.DataFrame(comparison_summary)
        comparison_df = comparison_df.set_index("Metrik").T
        st.dataframe(comparison_df)
    else:
        st.warning("⚠️ Tidak ada data untuk ditampilkan. Pastikan rentang waktu valid.")
        
    st.markdown("---")
    st.subheader("Plot Perbandingan Time Series")
    
    selected_models = st.multiselect(
        "Pilih model yang akan di-plot:",
        options=list(dataset_info.keys()),
        default=list(dataset_info.keys())
    )

    metrics_to_plot = st.multiselect(
        "Pilih metrik untuk di-plot:",
        options=['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'],
        default=['ch_pred', 'rainfall']
    )

    if not selected_models or not metrics_to_plot:
        st.info("💡 Pilih setidaknya satu model dan satu metrik untuk menampilkan plot.")
    else:
        # Pisahkan data rainfall dari metrik lainnya
        is_rainfall_selected = 'rainfall' in metrics_to_plot
        other_metrics = [m for m in metrics_to_plot if m != 'rainfall']

        dfs_to_plot = []
        
        # Ambil data rainfall dari salah satu model (semuanya sama)
        if is_rainfall_selected and selected_models:
            first_model = selected_models[0]
            df_rainfall = st.session_state.comparative_data.get(first_model)
            if not df_rainfall.empty:
                rainfall_df = df_rainfall[['year', 'month', 'rainfall']].copy()
                rainfall_df['model_name'] = 'Ground Truth' # Label khusus
                rainfall_df = rainfall_df.rename(columns={'rainfall': 'Value'})
                rainfall_df['Metric'] = 'rainfall'
                dfs_to_plot.append(rainfall_df)

        # Gabungkan data untuk metrik lainnya dari model yang dipilih
        for model_name in selected_models:
            df = st.session_state.comparative_data.get(model_name, pd.DataFrame())
            if not df.empty and other_metrics:
                df_other_metrics = df[['year', 'month'] + other_metrics].copy()
                df_other_metrics['model_name'] = model_name
                
                melted_df = df_other_metrics.melt(
                    id_vars=['year', 'month', 'model_name'],
                    value_vars=other_metrics,
                    var_name='Metric',
                    value_name='Value'
                )
                dfs_to_plot.append(melted_df)

        if not dfs_to_plot:
            st.warning("⚠️ Data tidak tersedia untuk model atau metrik yang dipilih.")
        else:
            combined_df = pd.concat(dfs_to_plot, ignore_index=True)
            combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1))
            
            # --- FIX: Mengurutkan DataFrame berdasarkan tanggal sebelum plotting ---
            combined_df.sort_values(by='date', inplace=True)
            # -------------------------------------------------------------------
            
            # Buat label baru untuk legend
            combined_df['combined_label'] = combined_df['Metric'] + ' (' + combined_df['model_name'] + ')'
            # Perbaiki label untuk rainfall
            combined_df.loc[combined_df['Metric'] == 'rainfall', 'combined_label'] = 'Rainfall (Ground Truth)'

            # Kamus (dictionary) untuk pemetaan warna
            color_map = {
                'Rainfall (Ground Truth)': 'saddlebrown',
                
                'ch_pred (0 Variabel)': 'royalblue',
                'error_bias (0 Variabel)': 'darkblue',
                'absolute_error (0 Variabel)': 'midnightblue',
                'squared_error (0 Variabel)': 'navy',
                
                'ch_pred (10 Variabel)': 'deeppink',
                'error_bias (10 Variabel)': 'darkred',
                'absolute_error (10 Variabel)': 'crimson',
                'squared_error (10 Variabel)': 'indianred',
                
                'ch_pred (51 Variabel)': 'forestgreen',
                'error_bias (51 Variabel)': 'darkgreen',
                'absolute_error (51 Variabel)': 'seagreen',
                'squared_error (51 Variabel)': 'olivedrab',
            }

            fig = px.line(
                combined_df,
                x='date',
                y='Value',
                color='combined_label',
                title=f'Perbandingan Time Series untuk Stasiun {selected_station_name}',
                labels={'Value': 'Nilai', 'date': 'Tanggal', 'combined_label': 'Metrik'},
                markers=True,
                color_discrete_map=color_map
            )
            
            st.plotly_chart(fig, use_container_width=True)

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