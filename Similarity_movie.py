import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("tmdb_5000_movies.csv")

movies = movies[['title', 'genres', 'keywords', 'overview']]

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

movies['overview'] = movies['overview'].fillna('')
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview']

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]

        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:11]

        print("\nTop 10 movies similar to:", movie)

        for i in movie_list:
            print(movies.iloc[i[0]].title)

    except IndexError:
        print("Movie not found in dataset.")

recommend("The Matrix")
recommend("Inception")
recommend("Interstellar")