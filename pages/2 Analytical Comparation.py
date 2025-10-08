# pages/Analytical Table.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np # Ditambahkan untuk perhitungan metrik
import matplotlib.pyplot as plt # Ditambahkan untuk Scatter Plot

st.set_page_config(
    page_title="Tabel Data Komparatif",
    page_icon="📊",
    layout="wide"
)

os.environ["STREAMLIT_WATCHDOG"] = "false"

# URL dasar untuk data prediksi
base_url_pred = "data/10k_epoch/pred"

# URL dasar untuk data padanan
base_url_padanan = "data/10k_epoch/padanan"

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

# --- FUNGSI BARU UNTUK MENGHITUNG METRIK ---
def calculate_metrics(df: pd.DataFrame, actual_col: str, pred_col: str):
    """
    Menghitung MAE, RMSE, dan R^2.
    Menggunakan kolom yang tersedia dan hanya baris tanpa NaN di kedua kolom.
    """
    # Pastikan hanya baris dengan nilai valid yang digunakan
    df_clean = df.dropna(subset=[actual_col, pred_col])
    
    if df_clean.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    actual = df_clean[actual_col].astype(float)
    pred = df_clean[pred_col].astype(float)

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - actual))

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((pred - actual)**2))

    # R^2 (Coefficient of Determination)
    ss_total = np.sum((actual - np.mean(actual))**2)
    ss_residual = np.sum((actual - pred)**2)
    
    # Handle zero division for ss_total in case of constant actual values
    if ss_total == 0:
        r2 = 1.0 # Perfect score if actual is constant and prediction matches
    else:
        r2 = 1 - (ss_residual / ss_total)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_monthly_comparative_bar_chart(tahun: int, selected_station_name: str):
    """
    Fungsi untuk menampilkan bar chart perbandingan curah hujan bulanan (prediksi vs ground truth) 
    dan di bawahnya, Scatter Plot MAE, RMSE, dan R^2.
    """
    st.markdown("---")
    st.subheader(f"Perbandingan Curah Hujan Bulanan dan Metrik ({tahun}) untuk Stasiun {selected_station_name}")

    station_info = next((s for s in station_data if s["name"] == selected_station_name), None)
    
    if not station_info:
        st.error("❌ Informasi stasiun tidak ditemukan.")
        return 
        
    all_data_for_plot = []
    all_combined_data = {} # Untuk menyimpan data mentah untuk metrik

    # 1. Ambil data Padanan (Ground Truth) dan Gabungkan dengan Prediksi
    df_padanan = load_padanan_data(tahun)
    df_padanan_station = df_padanan[ 
        (df_padanan['year'] == tahun) & 
        (df_padanan['latitude'] == station_info['lat']) &  
        (df_padanan['longitude'] == station_info['lon']) 
    ].copy() 

    # Rename kolom untuk Ground Truth untuk Bar Chart
    if not df_padanan_station.empty and 'rainfall' in df_padanan_station.columns: 
        df_padanan_plot = df_padanan_station.rename(columns={'rainfall': 'Curah Hujan (mm)'}) 
        df_padanan_plot['Tipe Data'] = 'Ground Truth (Rainfall)' 
        df_padanan_plot['Warna'] = 'Ground Truth (Rainfall)' 
        all_data_for_plot.append(df_padanan_plot[['month', 'Curah Hujan (mm)', 'Tipe Data', 'Warna']]) 
    else: 
        st.warning(f"⚠️ Ground Truth (Rainfall) tidak tersedia untuk tahun {tahun}.") 


    # 2. Ambil data Prediksi (0, 10, 51 Variabel)
    for dataset_name in dataset_info.keys(): 
        df_pred = load_data(dataset_name, tahun) 
        df_pred_station = df_pred[ 
            (df_pred['year'] == tahun) & 
            (df_pred['latitude'] == station_info['lat']) &  
            (df_pred['longitude'] == station_info['lon']) 
        ].copy() 
        
        # Gabungkan data Prediksi dan Aktual (Padanan) untuk perhitungan Metrik dan Scatter Plot
        df_merged_yearly = pd.merge(
            df_pred_station[['month', 'ch_pred']], 
            df_padanan_station[['month', 'rainfall']], 
            on='month', 
            how='inner' # Hanya bulan yang memiliki kedua data yang dipertimbangkan
        ).drop_duplicates(subset=['month'])

        all_combined_data[dataset_name] = df_merged_yearly
        
        # Data untuk Bar Chart
        if not df_pred_station.empty and 'ch_pred' in df_pred_station.columns: 
            df_pred_plot = df_pred_station.rename(columns={'ch_pred': 'Curah Hujan (mm)'}) 
            df_pred_plot['Tipe Data'] = f'Prediksi ({dataset_name})' 
            df_pred_plot['Warna'] = f'Prediksi ({dataset_name})' 
            all_data_for_plot.append(df_pred_plot[['month', 'Curah Hujan (mm)', 'Tipe Data', 'Warna']]) 
        else: 
            st.warning(f"⚠️ Prediksi ({dataset_name}) tidak tersedia untuk tahun {tahun}.") 

    # --- Plot Bar Chart (PLotly Express) ---
    if not all_data_for_plot: 
        st.error("❌ Tidak ada data (prediksi maupun ground truth) yang ditemukan untuk periode ini.") 
        return 

    df_plot = pd.concat(all_data_for_plot, ignore_index=True) 
    df_plot['Bulan'] = df_plot['month'].map(bulan_dict) 
    df_plot = df_plot.sort_values(by=['month', 'Tipe Data']) 

    bar_color_map = { 
        'Ground Truth (Rainfall)': 'saddlebrown', 
        'Prediksi (0 Variabel)': 'royalblue', 
        'Prediksi (10 Variabel)': 'deeppink', 
        'Prediksi (51 Variabel)': 'forestgreen' 
    } 

    fig_bar = px.bar( 
        df_plot,  
        x='Bulan',  
        y='Curah Hujan (mm)', 
        color='Warna', 
        barmode='group',
        color_discrete_map=bar_color_map, 
        title=f'Curah Hujan Bulanan Komparatif ({tahun}) di Stasiun {selected_station_name}', 
        labels={'Curah Hujan (mm)': 'Curah Hujan (mm)', 'Bulan': 'Bulan'}, 
    ) 
    
    fig_bar.update_layout( 
        xaxis_title="Bulan", 
        yaxis_title="Curah Hujan (mm)", 
        legend_title="Tipe Data", 
        bargap=0.15, 
        xaxis={'categoryorder':'array', 'categoryarray': [bulan_dict[m] for m in range(1, 13)]} 
    ) 
    
    st.plotly_chart(fig_bar, use_container_width=True) 
    # --- Akhir Plot Bar Chart ---

    st.markdown("---")
    st.subheader(f"Scatter Plot Curah Hujan Aktual vs Prediksi Bulanan ({tahun})")

    # --- Plot Scatter Plot (Matplotlib) ---
    
    # Tentukan layout plot: 3 kolom untuk 3 model
    fig_scatter, axes = plt.subplots(1, 3, figsize=(18, 6)) # 1 baris, 3 kolom
    plt.style.use('ggplot')
    
    scatter_color_map = {
        '0 Variabel': 'royalblue',
        '10 Variabel': 'deeppink',
        '51 Variabel': 'forestgreen'
    }
    
    i = 0
    max_val = 0 # Untuk menyesuaikan batas sumbu
    
    for dataset_name, df_combined in all_combined_data.items():
        ax = axes[i]
        
        # Hitung Metrik Tahunan
        metrics = calculate_metrics(df_combined, 'rainfall', 'ch_pred')
        
        # Data untuk Scatter Plot (1 titik = 1 bulan)
        actual = df_combined['rainfall'].astype(float)
        pred = df_combined['ch_pred'].astype(float)
        
        # Update max_val
        if not actual.empty and not pred.empty:
            current_max = max(actual.max(), pred.max())
            if current_max > max_val:
                max_val = current_max

        # Scatter Plot
        ax.scatter(actual, pred, color=scatter_color_map[dataset_name], label=dataset_name, alpha=0.7)
        
        # Garis 1:1 (Ideal)
        # Akan disesuaikan setelah mendapatkan max_val global
        
        # Teks Metrik di Pojok Kanan Atas
        textstr = '\n'.join((
            r'MAE = %.2f' % (metrics['MAE'], ),
            r'RMSE = %.2f' % (metrics['RMSE'], ),
            r'$R^2$ = %.2f' % (metrics['R2'], )))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.0) # Background transparan
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)


        # Label dan Judul
        ax.set_title(f'Model {dataset_name}', fontsize=14)
        ax.set_xlabel('Curah Hujan Aktual Bulanan (mm)', fontsize=12)
        ax.set_ylabel('Curah Hujan Prediksi Bulanan (mm)', fontsize=12)
        
        i += 1

    # Atur batas sumbu X dan Y agar sama, dan tambahkan garis 1:1
    # Tambahkan sedikit padding
    plot_limit = max_val * 1.05
    for ax in axes:
        ax.set_xlim(0, plot_limit)
        ax.set_ylim(0, plot_limit)
        # Garis 1:1
        ax.plot([0, plot_limit], [0, plot_limit], color='black', linestyle='--', linewidth=1)

    # Sesuaikan layout dan tampilkan plot di Streamlit
    plt.tight_layout()
    st.pyplot(fig_scatter)
    # --- Akhir Plot Scatter Plot ---

# Main Streamlit app logic 
st.title("📊 DK Viewer - Tabel Analisis Komparatif") 

if 'comparative_data' not in st.session_state: 
    st.session_state.comparative_data = None 
if 'combinations' not in st.session_state: 
    st.session_state.combinations = [] 
if 'selected_station_name' not in st.session_state: 
    st.session_state.selected_station_name = station_names[0] 

with st.sidebar.form("config_form"): 
    st.header("⚙️ Konfigurasi") 
    
    year_options = list(range(1985, 2015)) 
    bulan_options = list(range(1, 13)) 
    
    display_type = st.radio( 
        "Pilih Tampilan Data:", 
        ["Time Series & Summary", "Bar Chart dan Scatter Plot Tahunan"] 
    ) 
    
    # --- Konfigurasi Rentang Waktu --- 
    if display_type == "Time Series & Summary": 
        st.subheader("Dari") 
        col1, col2 = st.columns(2) 
        with col1: 
            bulan_from = st.selectbox( 
                "Bulan Awal:", 
                bulan_options, 
                index=0, 
                format_func=lambda x: bulan_dict[x], 
                key="bulan_from_ts" 
            ) 
        with col2: 
            tahun_from = st.selectbox("Tahun Awal:", year_options, index=0, key="tahun_from_ts") 
        
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
        
        tahun_map_bar_chart = None # Non-relevan 
        
    else: # display_type == "Bar Chart Komparatif Bulanan (1 Tahun)" 
        st.subheader("Pilih Tahun") 
        
        # Untuk Bar Chart Komparatif Bulanan, From dan Until harus di set ke 1 Januari - 31 Desember 
        bulan_from = 1 
        bulan_until = 12 
        
        tahun_map_bar_chart = st.selectbox("Tahun:", year_options, index=0, key="tahun_map") 
        tahun_from = tahun_map_bar_chart 
        tahun_until = tahun_map_bar_chart 
    
    selected_station_name = st.selectbox("Pilih stasiun:", station_names) 
    st.session_state.selected_station_name = selected_station_name 
    
    submit = st.form_submit_button("🚀 Submit konfigurasi dan bandingkan") 

if submit: 
    from_date = (tahun_from, bulan_from) 
    until_date = (tahun_until, bulan_until) 

    if from_date > until_date: 
        st.error("❌ Tanggal 'Dari' tidak boleh lebih baru dari tanggal 'Sampai'.") 
    elif display_type == "Bar Chart dan Scatter Plot Tahunan": 
        # Pastikan rentang waktu yang dipilih adalah 1 tahun penuh (Januari-Desember) 
        if tahun_from != tahun_until or bulan_from != 1 or bulan_until != 12: 
            st.warning("⚠️ Untuk Bar Chart Komparatif Bulanan, Anda harus memilih rentang waktu 1 tahun penuh (Januari hingga Desember) yang sama.") 
            # Tetap plot sesuai tahun yang dipilih di widget (tahun_map_bar_chart), asumsikan user ingin melihat 12 bulan dari tahun tersebut 
            plot_monthly_comparative_bar_chart(tahun_map_bar_chart, selected_station_name) 
            st.success("✅ Data berhasil dimuat dan siap untuk perbandingan Bar Chart Bulanan dan Scatter Plot.") 
        else: 
            st.session_state.comparative_data = {} # Reset data Time Series 
            plot_monthly_comparative_bar_chart(tahun_map_bar_chart, selected_station_name) 
            st.success("✅ Data berhasil dimuat dan siap untuk perbandingan Bar Chart Bulanan dan Scatter Plot.") 
    
    else: # Time Series & Summary (Logika lama)
        # Logika Time Series & Summary (tidak diubah dari kode asli)
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
                
                if 'rainfall' in df_filtered_all.columns: 
                    df_filtered_all['error_bias'] = df_filtered_all['ch_pred'] - df_filtered_all['rainfall'] 
                    df_filtered_all['absolute_error'] = abs(df_filtered_all['ch_pred'] - df_filtered_all['rainfall']) 
                    df_filtered_all['squared_error'] = (df_filtered_all['ch_pred'] - df_filtered_all['rainfall'])**2 
                else: 
                    df_filtered_all['error_bias'] = None 
                    df_filtered_all['absolute_error'] = None 
                    df_filtered_all['squared_error'] = None 


                station_info = next(s for s in station_data if s["name"] == selected_station_name) 
                df_filtered_station = df_filtered_all[ 
                    (df_filtered_all['latitude'] == station_info['lat']) &  
                    (df_filtered_all['longitude'] == station_info['lon']) 
                ].copy() 

                mask = ( 
                    (df_filtered_station['year'] > tahun_from) | 
                    ((df_filtered_station['year'] == tahun_from) & (df_filtered_station['month'] >= bulan_from)) 
                ) & ( 
                    (df_filtered_station['year'] < tahun_until) | 
                    ((df_filtered_station['year'] == tahun_until) & (df_filtered_station['month'] <= bulan_until)) 
                ) 
                df_filtered_station = df_filtered_station[mask].copy() 


                filtered_data_dict[dataset_name] = df_filtered_station 
            else: 
                filtered_data_dict[dataset_name] = pd.DataFrame() 

        st.session_state.comparative_data = filtered_data_dict 
        st.success("✅ Data berhasil dimuat dan siap untuk perbandingan.") 

# --- Tampilan Time Series & Summary (Tidak Berubah) --- 
if st.session_state.comparative_data and st.session_state.comparative_data.keys() and display_type == "Time Series & Summary": 
    
    # Ringkasan Statistik 
    st.markdown("---") 
    st.subheader(f"Ringkasan Statistik Komparatif untuk Stasiun {st.session_state.selected_station_name}") 
    
    summary_cols = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'] 
    comparison_summary = [] 
    
    for dataset_name, df in st.session_state.comparative_data.items(): 
        if not df.empty: 
            summary_row = {"Metrik": dataset_name} 
            for col in summary_cols: 
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): 
                    summary_row[f"Mean ({col})"] = df[col].mean() 
                    summary_row[f"Sum ({col})"] = df[col].sum() 
                else: 
                    summary_row[f"Mean ({col})"] = None 
                    summary_row[f"Sum ({col})"] = None 
            comparison_summary.append(summary_row) 

    if comparison_summary: 
        comparison_df = pd.DataFrame(comparison_summary).set_index("Metrik").T 
        for col in comparison_df.columns: 
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else None) 
            
        st.dataframe(comparison_df) 
    else: 
        st.warning("⚠️ Tidak ada data untuk ditampilkan. Pastikan rentang waktu valid.") 
        
    # Plot Time Series 
    st.markdown("---") 
    st.subheader("Plot Perbandingan Time Series") 
    
    selected_models = st.multiselect( 
        "Pilih model yang akan di-plot:", 
        options=list(dataset_info.keys()), 
        default=list(dataset_info.keys()), 
        key='ts_models' 
    ) 

    metrics_to_plot = st.multiselect( 
        "Pilih metrik untuk di-plot:", 
        options=['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'], 
        default=['ch_pred', 'rainfall'], 
        key='ts_metrics' 
    ) 

    if not selected_models or not metrics_to_plot: 
        st.info("💡 Pilih setidaknya satu model dan satu metrik untuk menampilkan plot.") 
    else: 
        is_rainfall_selected = 'rainfall' in metrics_to_plot 
        other_metrics = [m for m in metrics_to_plot if m != 'rainfall'] 

        dfs_to_plot = [] 
        
        # Logika penggabungan data untuk Time Series 
        if is_rainfall_selected and selected_models: 
            first_model = selected_models[0] 
            df_rainfall = st.session_state.comparative_data.get(first_model) 
            if not df_rainfall.empty and 'rainfall' in df_rainfall.columns: 
                rainfall_df = df_rainfall[['year', 'month', 'rainfall']].copy() 
                rainfall_df['model_name'] = 'Ground Truth' 
                rainfall_df = rainfall_df.rename(columns={'rainfall': 'Value'}) 
                rainfall_df['Metric'] = 'rainfall' 
                dfs_to_plot.append(rainfall_df) 

        for model_name in selected_models: 
            df = st.session_state.comparative_data.get(model_name, pd.DataFrame()) 
            if not df.empty and other_metrics: 
                existing_other_metrics = [m for m in other_metrics if m in df.columns] 
                
                if existing_other_metrics: 
                    df_other_metrics = df[['year', 'month'] + existing_other_metrics].copy() 
                    df_other_metrics['model_name'] = model_name 
                    
                    melted_df = df_other_metrics.melt( 
                        id_vars=['year', 'month', 'model_name'], 
                        value_vars=existing_other_metrics, 
                        var_name='Metric', 
                        value_name='Value' 
                    ) 
                    dfs_to_plot.append(melted_df) 

        if not dfs_to_plot: 
            st.warning("⚠️ Data tidak tersedia untuk model atau metrik yang dipilih.") 
        else: 
            combined_df = pd.concat(dfs_to_plot, ignore_index=True) 
            combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1)) 
            combined_df.sort_values(by='date', inplace=True) 
            combined_df['combined_label'] = combined_df['Metric'] + ' (' + combined_df['model_name'] + ')' 
            combined_df.loc[combined_df['Metric'] == 'rainfall', 'combined_label'] = 'Rainfall (Ground Truth)' 

            color_map = { 
                'Rainfall (Ground Truth)': 'saddlebrown', 'ch_pred (0 Variabel)': 'royalblue', 
                'error_bias (0 Variabel)': 'darkblue', 'absolute_error (0 Variabel)': 'midnightblue', 
                'squared_error (0 Variabel)': 'navy', 'ch_pred (10 Variabel)': 'deeppink', 
                'error_bias (10 Variabel)': 'darkred', 'absolute_error (10 Variabel)': 'crimson', 
                'squared_error (10 Variabel)': 'indianred', 'ch_pred (51 Variabel)': 'forestgreen', 
                'error_bias (51 Variabel)': 'darkgreen', 'absolute_error (51 Variabel)': 'seagreen', 
                'squared_error (51 Variabel)': 'olivedrab', 
            } 

            fig = px.line( 
                combined_df, 
                x='date', y='Value', color='combined_label', 
                title=f'Perbandingan Time Series untuk Stasiun {st.session_state.selected_station_name}', 
                labels={'Value': 'Nilai', 'date': 'Tanggal', 'combined_label': 'Metrik'}, 
                markers=True, color_discrete_map=color_map 
            ) 
            st.plotly_chart(fig, use_container_width=True) 

# --- Pesan Akhir (Tidak Berubah) --- 
elif not submit: 
    st.info("💡 Pilih Tampilan Data, rentang waktu/tahun, dan stasiun di sidebar, lalu tekan 'Submit konfigurasi dan bandingkan' untuk melihat data.") 

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