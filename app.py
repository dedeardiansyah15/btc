import streamlit as st
import joblib
import requests
import pandas as pd
from datetime import datetime, timedelta
import ta  # Untuk indikator teknikal
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Fungsi untuk mengambil data harga Bitcoin dari CoinGecko (harian)
def get_bitcoin_price_data(start_date, end_date):
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={start_timestamp}&to={end_timestamp}'
    response = requests.get(url)
    data = response.json()
    
    if 'prices' not in data:
        st.error("Error fetching Bitcoin data.")
        return pd.DataFrame()
    
    prices = data['prices']
    volumes = data['total_volumes']  # Ambil volume perdagangan Bitcoin
    
    # Convert to DataFrame
    df = pd.DataFrame(prices, columns=['timestamp', 'price_btc'])
    df['volume_btc'] = [volume[1] for volume in volumes]  # Ambil volume
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    return df[['date', 'price_btc', 'volume_btc']]

# Fungsi untuk mengambil data harga emas dari MetalPrice API
def get_gold_price_data(api_key):
    url = f'https://api.metalpriceapi.com/v1/latest?api_key={api_key}&base=USD&currencies=XAU'
    response = requests.get(url)
    
    if response.status_code != 200:
        st.error(f"Error fetching data: {response.text}")
        return pd.DataFrame()
    
    data = response.json()
    
    if 'rates' not in data:
        st.error("Error fetching Gold data from MetalPrice API.")
        return pd.DataFrame()

    gold_price = data['rates']['XAU']
    return pd.DataFrame([{'date': datetime.utcnow().date(), 'price_gold': gold_price}])

# Fungsi untuk menghitung indikator teknikal (RSI)
def calculate_rsi(price_data, window=14):
    price_data['RSI'] = ta.momentum.RSIIndicator(price_data['price_btc'], window=window).rsi().fillna(0)
    return price_data

# Fungsi untuk memuat model yang sudah disimpan dan memprediksi harga Bitcoin besok
def load_model(model_path):
    model = joblib.load(model_path)  # Gunakan joblib.load untuk .joblib
    return model

# Fungsi untuk prediksi harga Bitcoin besok
def predict_bitcoin_price(model, data):
    last_row = data.iloc[-1]
    X_last = np.array([[last_row['price_btc'], last_row['price_gold'], last_row['volume_btc'], last_row['RSI']]])  # Fitur terakhir
    predicted_price = model.predict(X_last)
    return predicted_price[0]

# Fungsi untuk menghitung persentase perubahan dan memberikan rekomendasi
def calculate_percentage_and_recommendation(current_price, predicted_price):
    percentage_change = ((predicted_price - current_price) / current_price) * 100
    if percentage_change > 0:
        recommendation = "Beli"  # Harga diprediksi naik
    else:
        recommendation = "Jangan beli"  # Harga diprediksi turun
    return percentage_change, recommendation

# Judul aplikasi Streamlit
st.title("📈 Prediksi Harga Bitcoin")

# Tentukan tanggal hari ini dan 1 bulan sebelumnya
end_date = datetime.utcnow()  # Hari ini
start_date = end_date - timedelta(days=30)  # 1 bulan yang lalu

# Masukkan API Key MetalPrice
API_KEY = '510066d3771b21d9653a365e44350b45'  # API Key yang kamu berikan

# Ambil data harga Bitcoin dan emas
with st.spinner('Mengambil data Bitcoin dan Emas...'):
    bitcoin_data = get_bitcoin_price_data(start_date, end_date)
    gold_data = get_gold_price_data(API_KEY)

# Tampilkan grafik harga Bitcoin dalam 1 bulan
st.subheader("📊 Grafik Harga Bitcoin (1 Bulan Terakhir)")
if not bitcoin_data.empty:
    fig = px.line(bitcoin_data, x='date', y='price_btc', title='Harga Bitcoin 1 Bulan Terakhir', labels={'price_btc': 'Harga Bitcoin (USD)', 'date': 'Tanggal'})
    fig.update_traces(line_color='blue', mode='lines+markers')
    fig.update_layout(xaxis_title='Tanggal', yaxis_title='Harga Bitcoin (USD)', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Data Bitcoin tidak tersedia.")

# Gabungkan data berdasarkan tanggal jika keduanya ada
if not bitcoin_data.empty and not gold_data.empty:
    merged_data = pd.merge(bitcoin_data, gold_data, on='date', how='inner')
    
    # Hitung RSI
    merged_data = calculate_rsi(merged_data)
    
    # Memuat model yang sudah disimpan (ganti dengan path model kamu)
    model_path = r'C:\Users\dedea\OneDrive\Desktop\Portofolio\Bitcoin\bitcoin_price_prediction_model.joblib'  # Ganti dengan path file model .joblib kamu
    
    model = load_model(model_path)
    
    # Prediksi harga Bitcoin besok
    predicted_price = predict_bitcoin_price(model, merged_data)
    current_price = merged_data.iloc[-1]['price_btc']  # Harga Bitcoin hari ini
    
    # Hitung persentase perubahan dan rekomendasi
    percentage_change, recommendation = calculate_percentage_and_recommendation(current_price, predicted_price)
    
    # Tampilkan hasil prediksi
    st.subheader("🔮 Prediksi Harga Bitcoin Besok")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Harga Bitcoin Hari Ini", value=f"${current_price:.2f}")
    
    with col2:
        st.metric(label="Prediksi Harga Besok", value=f"${predicted_price:.2f}", delta=f"{percentage_change:.2f}%")
    
    with col3:
        if percentage_change > 0:
            st.success(f"Rekomendasi: **{recommendation}** 🚀")
        else:
            st.error(f"Rekomendasi: **{recommendation}** ⚠️")
else:
    st.error("Data tidak tersedia atau tidak dapat digabungkan.")