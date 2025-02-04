import streamlit as st
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# ----------------------------- KONFIGURASI -----------------------------
# Tentukan periode hari untuk data (misalnya 30 hari)
DAYS = 30

# Muat file .env untuk mengambil API KEY
dotenv_path = 'API.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    st.error("File .env tidak ditemukan!")
    st.stop()

API_KEY = os.getenv('NEWS_API_KEY')
if not API_KEY:
    st.error("API Key tidak ditemukan di file .env!")
    st.stop()

# Muat model yang telah disimpan
try:
    model = joblib.load('bitcoin_price_prediction_model.joblib')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ----------------------------- FUNGSI UTAMA -----------------------------
# Fungsi untuk mengambil harga Bitcoin dan emas dari CoinGecko
def get_today_prices():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin,pax-gold',
        'vs_currencies': 'usd'
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        btc_price = data.get('bitcoin', {}).get('usd', None)
        gold_price = data.get('pax-gold', {}).get('usd', None)
        return btc_price, gold_price
    except Exception as e:
        st.error(f"Gagal mengambil data harga: {e}")
        return None, None

# Fungsi untuk mengambil data berita dari NewsAPI
def get_news_data(query='Bitcoin', language='en', days=DAYS):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    url = (
        f'https://newsapi.org/v2/everything?q={query}'
        f'&language={language}&from={start_date.date()}&to={end_date.date()}'
        f'&apiKey={API_KEY}'
    )
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        st.error(f"Gagal mengambil data berita. Status code: {response.status_code}")
        return []

# Fungsi untuk menganalisis sentimen menggunakan VADER
def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        text = f"{title} {description} {content}"
        sentiment_score = analyzer.polarity_scores(text)['compound']
        sentiments.append(sentiment_score)
    # Jika ada beberapa artikel, ambil rata-rata nilai sentimen
    if sentiments:
        return np.mean(sentiments)
    else:
        return 0.0  # Netral jika tidak ada data

# Fungsi untuk mengambil data harga Bitcoin selama periode tertentu
def get_bitcoin_price_data(days=DAYS):
    btc_data = yf.download('BTC-USD', period=f'{days}d', interval='1d')
    btc_data = btc_data[['Close']].reset_index()
    btc_data['date'] = btc_data['Date'].dt.date
    return btc_data[['date', 'Close']]

# Fungsi untuk mengambil data harga emas selama periode tertentu
def get_gold_price_data(days=DAYS):
    gold_data = yf.download('GC=F', period=f'{days}d', interval='1d')
    gold_data = gold_data[['Close']].reset_index()
    gold_data['date'] = gold_data['Date'].dt.date
    return gold_data[['date', 'Close']]

# Fungsi untuk menghitung nilai RSI dari data harga Bitcoin
def calculate_rsi(btc_data, window=14):
    btc_data['Close'] = btc_data['Close'].astype(float)
    # Pastikan data harga dalam bentuk 1D dengan squeeze()
    close_series = btc_data['Close'].squeeze()
    rsi_indicator = ta.momentum.RSIIndicator(close_series, window=window)
    btc_data['RSI'] = rsi_indicator.rsi().fillna(0)
    return btc_data['RSI'].iloc[-1]

# ----------------------------- CSS Styling -----------------------------
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            color: #ffffff;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            background: linear-gradient(145deg, #6a11cb, #2575fc);
            border-radius: 10px;
            padding: 15px;
        }
        .header {
            font-size: 24px;
            color: #ffffff;
            margin-top: 20px;
            text-align: center;
        }
        .card {
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .recommendation {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .error {
            font-size: 18px;
            color: #dc3545;
        }
        .info {
            font-size: 16px;
            margin-top: 20px;
            text-align: center;
        }
        .stButton>button {
            background: linear-gradient(145deg, #6a11cb, #2575fc);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(145deg, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .container {
            background: #212529;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
            max-width: 900px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- APLIKASI UTAMA -----------------------------
# Ambil data harga hari ini dari CoinGecko
btc_price, gold_price = get_today_prices()

if btc_price is None or gold_price is None:
    st.markdown("<p class='error'>Gagal mengambil data harga dari CoinGecko.</p>", unsafe_allow_html=True)
    st.stop()

# Ambil data berita dan hitung rata-rata sentimen
articles = get_news_data(query='Bitcoin', days=DAYS)
sentiment = analyze_sentiment(articles)

# Ambil data harga Bitcoin (untuk menghitung RSI) selama periode DAYS
btc_data = get_bitcoin_price_data(days=DAYS)
rsi = calculate_rsi(btc_data)

# Tentukan warna box untuk setiap data
warna_btc   = "#F7931A"  # DodgerBlue untuk harga Bitcoin
warna_gold  = "#FFD700"  # Gold untuk harga emas
warna_sent  = "#6A5ACD"  # SlateBlue untuk sentimen
warna_rsi   = "#FF69B4"  # HotPink untuk RSI

# Siapkan warna untuk box prediksi berdasarkan kondisi
if (( (model.predict(np.array([[sentiment, btc_price, gold_price, rsi]]) )[0] - btc_price) / btc_price)*100) > 2:
    warna_prediksi = "#28a745"  # Hijau
elif (( (model.predict(np.array([[sentiment, btc_price, gold_price, rsi]]) )[0] - btc_price) / btc_price)*100) < -2:
    warna_prediksi = "#dc3545"  # Merah
else:
    warna_prediksi = "#ffc107"  # Kuning

# Siapkan fitur untuk prediksi
fitur = np.array([[sentiment, btc_price, gold_price, rsi]])

# Lakukan prediksi harga Bitcoin besok
predicted_price = model.predict(fitur)[0]
price_change_percent = ((predicted_price - btc_price) / btc_price) * 100

# Tentukan rekomendasi (juga untuk teks rekomendasi)
if price_change_percent > 2:
    recommendation = "<strong>Saran: Beli</strong> (Harga Prediksi Naik > 2%)"
    recommendation_color = "#28a745"
elif price_change_percent < -2:
    recommendation = "<strong>Saran: Jual</strong> (Harga Prediksi Turun < -2%)"
    recommendation_color = "#dc3545"
else:
    recommendation = "<strong>Saran: Tahan</strong> (Perubahan Harga < 2%)"
    recommendation_color = "#ffc107"

# ----------------------------- TAMPILAN -----------------------------
with st.container():
    # Judul aplikasi
    st.markdown("<p class='title'>Prediksi Harga Bitcoin Besok</p>", unsafe_allow_html=True)
    st.write(f"**Tanggal:** {datetime.today().strftime('%Y-%m-%d')}")
    
    # Tampilkan data harga Bitcoin dan Emas dengan warna box masing-masing
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='card' style='background-color: {warna_btc};'><h3>Harga Bitcoin</h3>"
            f"<p style='font-size: 24px;'>${btc_price}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div class='card' style='background-color: {warna_gold};'><h3>Harga Emas (PAX Gold)</h3>"
            f"<p style='font-size: 24px;'>${gold_price}</p></div>", unsafe_allow_html=True)
    
    # Tampilkan data Sentimen dan RSI dengan warna box masing-masing
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"<div class='card' style='background-color: {warna_sent};'><h3>Sentimen</h3>"
            f"<p style='font-size: 24px;'>{sentiment:.4f}</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(
            f"<div class='card' style='background-color: {warna_rsi};'><h3>RSI</h3>"
            f"<p style='font-size: 24px;'>{rsi:.2f}</p></div>", unsafe_allow_html=True)
    
    # Tampilkan hasil prediksi dengan box berwarna dinamis
    st.markdown("<p class='header'>Hasil Prediksi</p>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card' style='background-color: {warna_prediksi};'>"
        f"<h3>Prediksi Harga Bitcoin Besok</h3>"
        f"<p style='font-size: 24px;'>${predicted_price:.2f}</p></div>", unsafe_allow_html=True)
    st.write(f"Persentase Perubahan Harga: **{price_change_percent:.2f}%**")
    st.markdown(f"<p class='recommendation' style='color:{recommendation_color};'>{recommendation}</p>", unsafe_allow_html=True)
    
    # Tampilkan grafik harga Bitcoin 1 bulan terakhir (menggunakan data dari yfinance)
    st.subheader("Grafik Harga Bitcoin 1 Bulan Terakhir")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Gunakan btc_data yang sudah diambil dari yfinance
    ax.plot(btc_data['date'], btc_data['Close'], label='Harga Bitcoin (USD)', color='blue', linewidth=2)
    
    # Format sumbu X agar tanggal lebih mudah dibaca
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Bitcoin (USD)')
    ax.set_title('Harga Bitcoin selama 1 bulan terakhir')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("<p class='info'>Prediksi ini dibuat dengan model Random Forest yang dilatih menggunakan data historis, sentimen, dan indikator teknikal (RSI). Pastikan untuk selalu memverifikasi informasi pasar terkini.</p>", unsafe_allow_html=True)
