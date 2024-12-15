import streamlit as st
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import Counter
 
# Membuat Dask Client untuk eksekusi terdistribusi
client = Client("tcp://172.16.12.189:8786")
 
# URL dataset
url = 'https://raw.githubusercontent.com/rrayhka/test/refs/heads/main/movies.csv'
 
# Membaca data menggunakan Dask
df = dd.read_csv(url)
 
# Mengambil genre dan judul dari DataFrame
df = df.compute()
judul_movie = df['title'].tolist()
genre_movie = df['genres'].tolist()
 
# Membuat DataFrame Pandas dari genre dan judul
data_pandas = pd.DataFrame({
    'judul': judul_movie,
    'genre': genre_movie
})
 
# Menghitung genre counts
genre_counts = data_pandas['genre'].value_counts().reset_index()
genre_counts.columns = ['genre', 'count']
 
# Menampilkan genre counts
st.write("Genre Counts", genre_counts)
 
# Menggunakan CountVectorizer untuk menghitung frekuensi kata pada genre
tf = CountVectorizer()
tfidf_matrix = tf.fit_transform(data_pandas['genre'])  # Menggunakan genre yang telah dihimpun
cosine_sim = cosine_similarity(tfidf_matrix)  # Menghitung cosine similarity antar film berdasarkan genre
 
# Menyimpan cosine similarity dalam DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=data_pandas['judul'], columns=data_pandas['judul'])
 
# Membuat indeks untuk pencarian film
indices = pd.Series(data_pandas.index, index=data_pandas['judul']).drop_duplicates()
 
# Fungsi untuk memberikan rekomendasi film berdasarkan cosine similarity
def movie_recommendations(judul, cosine_sim):
    # Menggunakan .loc untuk mengakses berdasarkan judul film
    idx = indices.loc[judul]  # Menggunakan loc untuk mencari indeks berdasarkan judul
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:20]  # Menyaring hasil yang relevan
    movie_indices = [i[0] for i in sim_scores]
 
    # Kembalikan hanya judul film yang relevan
    return data_pandas.loc[movie_indices, 'judul'].tolist()
 
# Menambahkan input untuk memilih film di Streamlit
selected_movie = st.selectbox('Pilih Film', judul_movie)
 
# Memberikan rekomendasi untuk film yang dipilih
if selected_movie:
    recomendation = movie_recommendations(selected_movie, cosine_sim)
    st.write(f"Rekomendasi untuk film **{selected_movie}**:")
    for r in recomendation:
        st.write(r)
 
# Menutup client Dask setelah selesai (tidak diperlukan dalam aplikasi ini)
# client.close()
