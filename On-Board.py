# main.py
import streamlit as st

st.set_page_config(
    page_title="DK Viewer",
    page_icon="🗺️",
)

st.title("Selamat Datang di Aplikasi DK Viewer!")
st.write("Aplikasi ini dibuat untuk melihat data prediksi curah hujan di berbagai wilayah")
st.write("**TRAIN 10k epoch**")
st.markdown("""
Silakan pilih halaman dari menu di **sidebar** untuk mulai menjelajahi data:

* **Fitur - Analytical Table**:  
  Menyediakan tabel analitis berisi data curah hujan historis maupun prediksi. Dilengkapi dengan filter stasiun (berdasarkan koordinat tertentu) agar pengguna dapat fokus pada lokasi spesifik. Selain itu, tersedia visualisasi *timeseries plot* untuk melihat tren curah hujan dari waktu ke waktu, memudahkan identifikasi pola musiman maupun anomali.  

* **Fitur - Analytical Comparation**:  
  Memungkinkan pengguna untuk melakukan perbandingan data antar-sumber (misalnya hasil prediksi dengan data observasi) secara kuantitatif. Analisis dapat berupa perhitungan metrik kesalahan (MAE, RMSE), deviasi antar dataset, maupun grafik komparasi sehingga memudahkan evaluasi performa model atau sumber data.  

* **Fitur - Hovmöller Comparation**:  
  Menyajikan metode visualisasi Hovmöller Diagram yang memperlihatkan distribusi curah hujan berdasarkan dimensi ruang (lintang/bujur) dan waktu. Fitur ini memungkinkan pengguna untuk membandingkan dinamika spasial-temporal antara hasil prediksi dan data observasi, sehingga perbedaan pola spasial dapat teridentifikasi dengan jelas.  

* **Fitur - Monthly Analytic**:  
  Memungkinkan perbandingan kuantitatif data antar-sumber yang berfokus pada satu bulan spesifik sepanjang tahun. Fitur ini menyediakan tabel statistik metrik kesalahan (Error Bias, Absolute Error, Squared Error) dan grafik Time Series tahunan untuk evaluasi pola kinerja musiman model.

* **Fitur - Map Viewer**:  
  Menyediakan peta interaktif yang menampilkan distribusi spasial curah hujan pada wilayah kajian. Peta ini dilengkapi dengan opsi *zoom in/out*, filter periode, serta interaksi klik pada titik koordinat untuk menampilkan nilai curah hujan. Dengan fitur ini, pengguna dapat melakukan eksplorasi visual secara intuitif dan memahami variasi curah hujan antar lokasi.  
---
""")


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
