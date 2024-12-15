"""# 2. Mengimpor pustaka/modul python yang dibutuhkan"""

import pandas as pd
import numpy as np 

import re
import string

import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.graph_objects as go
import plotly.express as px
# from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import plotly.io as pio
pio.renderers.default = 'colab'
from wordcloud import WordCloud,STOPWORDS

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

"""# 3. Data Understanding

## 3.1 Menyiapkan path dataset pada penyimpanan drive serta menampilkan overview dataset Movie menggunakan library pandas
"""

movies_meta_data = pd.read_csv('movies.csv')

movies_meta_data

"""### 3.2 Menampilkan keterangan jumlah/panjang data unique daftar film dan data pengguna/user"""

print('Banyak Data movies ID: ', len(movies_meta_data.movieId.unique()))

"""### 3.3 Menampilkan keterangan kolom dataset"""

# Memuat informasi dataframe movies_meta_data
movies_meta_data.info()

"""### 3.4 Menampilkan Daftar Genre pada dataset"""

print('Jenis-jenis Genre pada dataset: ', movies_meta_data.genres.unique())

"""### 3.5 menghitung besar/panjang data genre secara unique"""

print('Jumlah data genre: ', len(movies_meta_data.genres.unique()))

"""### 3.6 Memuat deskripsi setiap kolom dataframe untuk perhitungan count, rata-rata, minimal value dan maximal value, dll"""

# Memuat deskripsi setiap kolom dataframe
movies_meta_data.describe()

# Menghitung jumlah data kosong pada setiap kolom
movies_meta_data.isnull().sum()

"""### 3.7 Memuat dataset ke dalam variable baru"""

# Memuat dataset ke dalam variable baru
movie = movies_meta_data.movieId.unique()

# Mengurutkan data dan menghapus data yang sama
movie = np.sort(np.unique(movie))

print('Jumlah seluruh data movie berdasarkan movieId: ', len(movie))

movie_info = pd.concat([movies_meta_data])

movie_info

"""### 3.8 Menampilkan jumlah kata paling banyak yg muncul dalam kolom genre"""

word_could_dict = Counter(movies_meta_data['genres'].tolist())
wordcloud = WordCloud(width = 2000, height = 1000).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

"""# 4. Data Preparation

### 4.1 Memilih kolom berdasarkan data yang dibutuhkan untuk melakukan content based learning berdasarkan genre yaitu judul dan genre
"""

judul_movie = movies_meta_data['title'].tolist()
genre_movie = movies_meta_data['genres'].tolist()

print(len(judul_movie))
print(len(genre_movie))

"""### 4.2 Membuat data menjadi dalam bentuk dataframe sehingga mudah untuk dipersiapkan"""

data = pd.DataFrame({
    'judul': judul_movie,
    'genre': genre_movie
})

data

# melihat informasi kolom pada data
data.info()

"""### 4.3 Memuat banyak data dari setiap unique value berdasarkan genre"""

value_genre = pd.DataFrame(data['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
print(len(value_genre))
pd.options.display.max_colwidth = 500
value_genre

# membuat data string tanda strip '-' pada variable data dihapus
data = data[data.genre != '-']

"""### Melihat kembali Jenis-Jenis Genre yang terdapat pada data"""

data.genre.unique()

"""### 4.3 Melakukan drop pada judul film yg double, dan berhasil menghapus beberapa judul"""

data = data.drop_duplicates('judul')
len(data)

"""### 4.4 Mmelakukan indeks ulang pada data agar penomoran dilakukan berurutan"""

data.reset_index()
data

"""### 4.5 Memasukkan nilai data masing-masing kolom ke dalam variabel baru"""

judul = data['judul'].tolist()
genre = data['genre'].tolist()

print(len(judul))
print(len(genre))

# mengecek ulang data yg dimasukkan ke dalam variable baru
data = pd.DataFrame({
    'judul': judul,
    'genre': genre
})
data

"""## 4.6 Proses Data

### 4.6.1 Membangun sistem rekomendasi berdasarkan genre yang ada pada setiap movies.
"""

# Inisialisasi CountVectorizer
tf = CountVectorizer()
 
# Melakukan perhitungan idf pada data genre
tf.fit(genre) 

# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

"""### 4.6.2 Melakukan Proses fit dan melihat jumlah ukuran matrix"""

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(genre) 
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

"""### 4.6.3 Mengubah vektor ke dalam bentuk matrix"""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

"""### 4.6.4 Melihat Daftar jumlah film berdasarkan genre dan melihat korelasi nya yg diperlihatkan dalam bentuk matrix"""

pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names_out(),
    index=data.judul
).sample(22, axis=1).sample(10, axis=0)

"""# 5 Modeling

### 5.1 Melatih Model dengan cosine similarity
"""

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

"""### 5.2 tahap ini menampilkan matriks kesamaan setiap judul dengan menampilkan judul film dalam 10 sampel kolom (axis = 1) dan 10 sampel baris (axis=0)."""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['judul'], columns=genre)
print('Shape:', cosine_sim_df.shape)
 

cosine_sim_df.sample(10, axis=1).sample(10, axis=0)

"""# 6. Evaluasi Model

### 6.1 Pada tahap ini dilakukan indikasi dan diperlihatkan judul film berdasarkan urutan dari data
"""

indices = pd.Series(index = data['judul'], data = data.index).drop_duplicates()
indices.head()

"""### 6.2 Membuat fungsi untuk memanggil 20 rekomendasi film berdasarkan judul yang di input"""

def movie_recommendations(judul, cosine_sim = cosine_sim,items=data[['judul','genre']]):
    # Mengambil indeks dari judul film yang telah didefinisikan sebelumnnya
    idx = indices[judul]
    
    # Mengambil skor kemiripan dengan semua judul film 
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Mengurutkan film berdasarkan skor kemiripan
    sim_scores = sorted(sim_scores, key = lambda x : x[1], reverse = True)
    
    # Mengambil 20 skor kemiripan dari 1-20 karena urutan 0 memberikan indeks yang sama dengan judul film yang diinput
    sim_scores = sim_scores[1:20]
    
    # Mengambil judul film dari skor kemiripan
    movie_indices = [i[0] for i in sim_scores]
    
    # Mengembalikan 20 rekomendasi judul film dari kemiripan skor yang telah diurutkan dan menampilkan genre dari 20 rekomendasi film tersebut
    return pd.DataFrame(data['judul'][movie_indices]).merge(items)

# mengecek judul film di dalam data
data[data.judul.eq('Johnny English Reborn (2011)')]

"""## 6.3 Mencoba menampilkan 19 rekomendasi film dari judul yang telah di input menggunakan fungsi movie_recomendations"""

recomendation = pd.DataFrame(movie_recommendations('Johnny English Reborn (2011)'))
recomendation

# menghitung banyaknya data genre pada hasil rekomendasi yg dilakukan 
value = pd.DataFrame(recomendation['genre'].value_counts().reset_index().values, columns = ['genre', 'count'])
value.head()

"""### 6.4 Melakukan perhitungan dengan menggunakan metrik precision untuk melihat akurasi"""

TP = 19 #jumlah prediksi benar untuk genre yang mirip atau serupa
FP = 0 #jumlah prediksi salah untuk genre yang mirip atau serupa

Precision = TP/(TP+FP)
print("{0:.0%}".format(Precision))