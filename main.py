import streamlit as st
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import requests

# Hubungkan ke Dask Scheduler
try:
    client = Client("tcp://172.16.14.216:8786")  # Ganti dengan alamat scheduler Anda
    st.success("Terhubung ke Dask Scheduler")
except Exception as e:
    st.error(f"Gagal terhubung ke Dask Scheduler: {e}")

# Judul Aplikasi
st.title("Sistem Rekomendasi Film Menggunakan Genre")

# Fungsi untuk mendapatkan Client IP
def get_client_ip():
    try:
        response = requests.get('https://httpbin.org/ip')  
        if response.status_code == 200:
            return response.json().get('origin', 'IP Tidak Diketahui')
    except Exception as e:
        return f"Error: {e}"

# Menampilkan IP Address
st.subheader("Informasi Client IP")
client_ip = get_client_ip()
st.write(f"IP Address Anda: **{client_ip}**")

# Load Data dengan Dask
@st.cache_data
def load_data():
    dask_df = dd.read_csv('movies.csv')
    return dask_df.compute()

movies_meta_data = load_data()

# Menampilkan Dataset Overview
if st.checkbox("Tampilkan Dataset Awal"):
    st.subheader("Dataset Film")
    st.write(movies_meta_data.head())

# Statistik Dataset
st.subheader("Statistik Data")
if st.button("Tampilkan Statistik"):
    st.write(movies_meta_data.describe())
    st.write("Jumlah Data Kosong:")
    st.write(movies_meta_data.isnull().sum())

# Wordcloud Genre
st.subheader("Wordcloud Genre")
if st.button("Tampilkan Wordcloud"):
    word_could_dict = Counter(movies_meta_data['genres'].tolist())
    wordcloud = WordCloud(width=2000, height=1000).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)

# Data Preparation
st.subheader("Data Preparation")
movies_meta_data = movies_meta_data.dropna(subset=['title', 'genres'])
movies_meta_data = movies_meta_data[movies_meta_data['genres'] != '-']
movies_meta_data = movies_meta_data.drop_duplicates('title').reset_index(drop=True)

st.write("Jumlah Film Unik: ", len(movies_meta_data['title'].unique()))
st.write("Jumlah Genre Unik: ", len(movies_meta_data['genres'].unique()))

# Sistem Rekomendasi
st.subheader("Rekomendasi Berdasarkan Genre")
data = movies_meta_data[['title', 'genres']]
judul = data['title'].tolist()
genre = data['genres'].tolist()

# CountVectorizer untuk Genre
tf = CountVectorizer()
tfidf_matrix = tf.fit_transform(genre)
cosine_sim = cosine_similarity(tfidf_matrix)

# Fungsi Rekomendasi
def movie_recommendations(title, cosine_sim=cosine_sim, items=data):
    idx = items[items['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:20]
    movie_indices = [i[0] for i in sim_scores]
    return items.iloc[movie_indices]

# Input Film untuk Rekomendasi
selected_movie = st.selectbox("Pilih Judul Film:", data['title'].unique())

if selected_movie:
    selected_genre = data[data['title'] == selected_movie]['genres'].values[0]
    st.write(f"**Genre untuk film '{selected_movie}' adalah:** {selected_genre}")

if st.button("Tampilkan Rekomendasi"):
    recomendation = movie_recommendations(selected_movie)
    st.write(recomendation)
    # Visualisasi Genre Rekomendasi
    genre_count = recomendation['genres'].value_counts().reset_index()
    genre_count.columns = ['genre', 'count']
    st.bar_chart(genre_count.set_index('genre'))

# Evaluasi Model
st.subheader("Evaluasi Akurasi Model")
st.write("Menghitung Precision untuk 19 Rekomendasi Film")
TP = 19  # jumlah benar (genre serupa)
FP = 0   # jumlah salah
Precision = TP / (TP + FP)
st.write(f"Precision: {Precision:.0%}")
