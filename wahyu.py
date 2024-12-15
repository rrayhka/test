import dask.dataframe as dd
from dask.distributed import Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
 
client = Client("tcp://172.16.12.189:8786")
 
url = 'https://raw.githubusercontent.com/rrayhka/test/refs/heads/main/movies.csv'
 
df = dd.read_csv(url)
 
print(df.head())
print(len(df))
 
df = df.compute()
word_could_dict = Counter(df['genres'].tolist())
judul_movie = df['title'].tolist()
genre_movie = df['genres'].tolist()
 
import pandas as pd
data_pandas = pd.DataFrame({
    'judul': judul_movie,
    'genre': genre_movie
})
 
data = dd.from_pandas(data_pandas, npartitions=4)
print(data.head())
 
genre_counts = data['genre'].value_counts()
 
value_genre = genre_counts.compute().reset_index()
value_genre.columns = ['genre', 'count']
 
judul = data['judul'].compute().tolist()
genre = data['genre'].compute().tolist()
 
tf = CountVectorizer()
tf.fit(genre) 
print(tf.get_feature_names_out())
tfidf_matrix = tf.fit_transform(genre) 
print(tfidf_matrix.shape)
tfidf_matrix.todense()
cosine_sim = cosine_similarity(tfidf_matrix) 
print (cosine_sim)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['judul'], columns=genre)
print('Shape:', cosine_sim_df.shape)
 
cosine_sim = cosine_similarity(tfidf_matrix)
 
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['judul'], columns=data['judul'])
print('Shape:', cosine_sim_df.shape)
 
indices = pd.Series(data.index, index=data['judul']).drop_duplicates()
 
def movie_recommendations(judul, cosine_sim, items=data[['judul', 'genre']]):
    idx = indices.loc[judul] 
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:20] 
    movie_indices = [i[0] for i in sim_scores]
 
    return data.loc[movie_indices, 'judul'].compute().tolist()
 
recomendation = movie_recommendations('Johnny English Reborn (2011)', cosine_sim)
for r in recomendation:
    print (r)
 
print(len(judul_movie))
print(len(genre_movie))
print(client)
client.close()
