# pages/Analytical Table.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(
    page_title="Tabel Data",
    page_icon="📋",
    layout="wide"
)

os.environ["STREAMLIT_WATCHDOG"] = "false"

# URL dasar untuk data prediksi
base_url_pred = "data/100k_epoch/pred"

# URL dasar untuk data padanan
base_url_padanan = "data/100k_epoch/padanan"

# Info dataset hanya untuk data 'pred' yang memiliki subfolder
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
st.title("📊 DK Viewer - Analytical Table")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'combinations' not in st.session_state:
    st.session_state.combinations = []

with st.sidebar.form("config_form"):
    st.header("⚙️ Konfigurasi")
    dataset_choice = st.selectbox("Pilih dataset:", list(dataset_info.keys()))
    
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
    
    submit = st.form_submit_button("🚀 Submit konfigurasi")

if submit:
    from_date = (tahun_from, bulan_from)
    until_date = (tahun_until, bulan_until)

    if from_date > until_date:
        st.error("❌ Tanggal 'Dari' tidak boleh lebih baru dari tanggal 'Sampai'.")
    else:
        tahun_final = list(range(tahun_from, tahun_until + 1))
        
        all_filtered = []
        for th in tahun_final:
            df_main = load_data(dataset_choice, th)
            df_padanan = load_padanan_data(th)
            
            if not df_main.empty and not df_padanan.empty:
                df_merged_year = pd.merge(df_main, df_padanan, on=['month', 'year', 'latitude', 'longitude'], how='left')
                df_merged_year = df_merged_year.drop_duplicates(subset=['latitude', 'longitude', 'month', 'year'])
                all_filtered.append(df_merged_year)
            elif not df_main.empty:
                all_filtered.append(df_main)
        
        if not all_filtered:
            st.warning("⚠️ Tidak ada data sesuai filter.")
            st.session_state.data = None
            st.session_state.combinations = []
        else:
            df_filtered_all = pd.concat(all_filtered, ignore_index=True)
            
            df_filtered_all['error_bias'] = df_filtered_all['ch_pred'] - df_filtered_all['rainfall']
            df_filtered_all['absolute_error'] = abs(df_filtered_all['ch_pred'] - df_filtered_all['rainfall'])
            df_filtered_all['squared_error'] = (df_filtered_all['ch_pred'] - df_filtered_all['rainfall'])**2

            df_filtered_all['bulan_tahun'] = df_filtered_all['month'].map(bulan_dict) + ' ' + df_filtered_all['year'].astype(str)
            unique_combinations = df_filtered_all['bulan_tahun'].unique()
            
            sorted_combinations = sorted(unique_combinations, key=lambda x: (int(x.rsplit(' ', 1)[1]), [k for k, v in bulan_dict.items() if v == x.rsplit(' ', 1)[0]][0]))
            
            st.session_state.data = df_filtered_all
            st.session_state.combinations = sorted_combinations
            st.success("✅ Data berhasil dimuat. Silakan pilih dari menu dropdown.")
    
if st.session_state.data is not None and len(st.session_state.combinations) > 0:
    
    st.subheader("Pilih Rentang Bulan dan Tahun")
    
    col1, col2 = st.columns(2)
    with col1:
        start_combo = st.selectbox(
            "Bulan dan Tahun Awal:",
            st.session_state.combinations
        )
    with col2:
        end_combo = st.selectbox(
            "Bulan dan Tahun Akhir:",
            st.session_state.combinations,
            index=len(st.session_state.combinations) - 1
        )
    
    start_idx = st.session_state.combinations.index(start_combo)
    end_idx = st.session_state.combinations.index(end_combo)
    
    if start_idx > end_idx:
        st.warning("⚠️ Bulan/Tahun awal tidak boleh lebih besar dari Bulan/Tahun akhir.")
    else:
        selected_combinations = st.session_state.combinations[start_idx:end_idx + 1]
        
        filtered_df = pd.DataFrame()
        for combo in selected_combinations:
            selected_month_str, selected_year_str = combo.rsplit(' ', 1)
            selected_month_num = [key for key, val in bulan_dict.items() if val == selected_month_str][0]
            selected_year_num = int(selected_year_str)
            
            df_slice = st.session_state.data[
                (st.session_state.data['month'] == selected_month_num) & 
                (st.session_state.data['year'] == selected_year_num)
            ]
            
            filtered_df = pd.concat([filtered_df, df_slice])

        station_info = next(s for s in station_data if s["name"] == selected_station_name)
        filtered_df = filtered_df[
            (filtered_df['latitude'] == station_info['lat']) & 
            (filtered_df['longitude'] == station_info['lon'])
        ]
        
        if filtered_df.empty:
            st.warning("⚠️ Data tidak ditemukan untuk rentang yang dipilih.")
        else:
            
            st.markdown("---")
            st.subheader("Plot Time Series")

            # Widget multiselect untuk memilih metrik yang akan diplot
            metrics_to_plot = st.multiselect(
                "Pilih metrik untuk di-plot:",
                options=['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error'],
                default=['ch_pred', 'rainfall'] # Nilai default saat pertama kali dimuat
            )

            if not metrics_to_plot:
                st.info("💡 Pilih setidaknya satu metrik untuk menampilkan plot.")
            else:
                # Siapkan data untuk plotting
                plot_df = filtered_df.copy()
                plot_df['date'] = pd.to_datetime(plot_df[['year', 'month']].assign(day=1))
                
                # Lelehkan DataFrame hanya untuk metrik yang dipilih
                melted_df = plot_df.melt(
                    id_vars=['date'], 
                    value_vars=metrics_to_plot, 
                    var_name='Metric', 
                    value_name='Value'
                )

                fig = px.line(
                    melted_df,
                    x='date',
                    y='Value',
                    color='Metric',
                    title=f'Plot Time Series untuk {selected_station_name}',
                    labels={'Value': 'Nilai', 'date': 'Tanggal', 'Metric': 'Metrik'},
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Ringkasan Statistik")

            summary_cols = ['ch_pred', 'rainfall', 'error_bias', 'absolute_error', 'squared_error']
            
            summary_data = {
                "Mean": [],
                "Sum": []
            }
            
            for col in summary_cols:
                if col in filtered_df.columns:
                    summary_data["Mean"].append(filtered_df[col].mean())
                    summary_data["Sum"].append(filtered_df[col].sum())
                else:
                    summary_data["Mean"].append(None)
                    summary_data["Sum"].append(None)

            summary_df = pd.DataFrame(summary_data, index=summary_cols)
            st.dataframe(summary_df)
            
            st.markdown("---")
            st.subheader("Tabel Data")

            df_display = filtered_df.drop(columns=['bulan_tahun'])
            
            new_column_order = [
                'latitude', 
                'longitude', 
                'month', 
                'year', 
                'ch_pred', 
                'rainfall', 
                'error_bias', 
                'absolute_error', 
                'squared_error', 
                'idx'
            ]
            
            cols_to_display = [col for col in new_column_order if col in df_display.columns]

            st.success(f"✅ Menampilkan data untuk {selected_station_name} dari {start_combo} sampai {end_combo} ({len(filtered_df)} baris)")
            st.dataframe(df_display[cols_to_display], height=500, width=1600)

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