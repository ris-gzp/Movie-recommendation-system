import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("E:\PROJECTS\ML PROJECTS\MOVIE RECOMMENDATION SYSTEM\movies_metadata.csv", low_memory=False)
credits = pd.read_csv("E:\PROJECTS\ML PROJECTS\MOVIE RECOMMENDATION SYSTEM\credits.csv")
keywords = pd.read_csv("E:\PROJECTS\ML PROJECTS\MOVIE RECOMMENDATION SYSTEM\keywords.csv")


movies['id'] = movies['id'].astype(str)
credits['id'] = credits['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)


movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')


movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


movies.dropna(inplace=True)


def convert(obj):
    """Convert stringified JSON lists into Python lists of names"""
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

def get_director(obj):
    """Extract director from crew"""
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
    except:
        return []
    return []

def get_top_cast(obj):
    """Extract top 3 actors"""
    try:
        L = []
        count = 0
        for i in ast.literal_eval(obj):
            if count < 3:
                L.append(i['name'])
                count += 1
        return L
    except:
        return []


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(get_top_cast)
movies['crew'] = movies['crew'].apply(get_director)


movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])


movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


new_df = movies[['id', 'title', 'tags']]


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


similarity = cosine_similarity(vectors)


def recommend(movie):
    if movie not in new_df['title'].values:
        print("❌ Movie not found in dataset.")
        return
    index = new_df[new_df['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\n🎬 Recommended movies for '{movie}':\n")
    for i in movies_list:
        print(f"👉 {new_df.iloc[i[0]].title}"
if __name__ == "__main__":
    movie_name = input("Enter a movie name: ")
    recommend(movie_name)
