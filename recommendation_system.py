import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv()

data = pd.read_csv("dataset/data.csv")

# Converter a coluna 'year' para numérico
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['year'] = data['year'].fillna(data['year'].mean())

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


from sklearn.cluster import KMeans
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data[number_cols].select_dtypes(np.number)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"]))

def find_song(name=None, artist=None):
    song_data = {}
    if name:
        results = sp.search(q=f'track:{name}', limit=1)
    elif artist:
        results = sp.search(q=f'artist:{artist}', limit=1)
    else:
        return None
    if results['tracks']['items'] == []:
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = results['name']
    song_data['artists'] = results['artists'][0]['name']  # Corrigido para 'artists'
    song_data['explicit'] = int(results['explicit'])
    song_data['duration_ms'] = results['duration_ms']
    song_data['popularity'] = results['popularity']
    song_data['year'] = int(results['album']['release_date'][:4])  # Convertido para inteiro

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame([song_data])


def get_song_data(song, spotify_data):
    if 'name' in song:
        try:
            song_data = spotify_data[spotify_data['name'] == song['name']].iloc[0]
            return song_data
        except IndexError:
            return find_song(name=song['name'])
    elif 'artist' in song:
        try:
            song_data = spotify_data[spotify_data['artists'].apply(lambda x: song['artist'].lower() in x.lower())]
            return song_data
        except IndexError:
            return find_song(artist=song['artist'])
    else:
        return None

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None or song_data.empty:
            return None 
        try:
            if isinstance(song_data, pd.DataFrame):
                song_vector = song_data[number_cols].mean().values
            else:
                song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)
        except KeyError:
            continue
    if len(song_vectors) == 0:
        return None
    return np.mean(np.array(song_vectors), axis=0)

def recommend_songs(song_list, spotify_data, n_songs=6):
    metadata_cols = ['artists', 'name', 'year']
    song_center = get_mean_vector(song_list, spotify_data)
    
    if song_center is None:
        return None  
    
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    
    df_rec_songs = rec_songs[metadata_cols].to_dict(orient='records')
    df_resultados = pd.DataFrame(df_rec_songs)
    
    df_resultados = df_resultados.rename(columns={
        'artists': 'Artista/Banda',
        'name': 'Musica',
        'year': 'Ano'
    })
    
    return df_resultados

def main():
    st.title("Sistema de Recomendação de Músicas com dados do Spotify")
    
    search_option = st.radio("Escolha o tipo de pesquisa:", ("Por Música", "Por Banda/Artista"))
    
    if search_option == "Por Música":
        song_name = st.text_input("Digite o nome da música:")
        if song_name:
            st.write(f"Buscando recomendações para a música: {song_name}")
            resultados = recommend_songs([{'name': song_name}], data)
            if resultados is None or resultados.empty:
                st.write(f"A música '{song_name}' não foi encontrada.")
            else:
                st.write(resultados)
    
    elif search_option == "Por Banda/Artista":
        artist_name = st.text_input("Digite o nome da banda ou artista:")
        if artist_name:
            st.write(f"Buscando recomendações para a banda/artista: {artist_name}")
            resultados = recommend_songs([{'artist': artist_name}], data)
            if resultados is None or resultados.empty:
                st.write(f"O artista/banda '{artist_name}' não foi encontrado.")
            else:
                st.write(resultados)

if __name__ == "__main__":
    main()
