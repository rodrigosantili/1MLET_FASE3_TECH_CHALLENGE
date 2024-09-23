import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clients import SpotifyClient
from src.data.data_handler import DataHandler


class SpotifyRecommendationSystem:
    def __init__(self, spotify_client: SpotifyClient, data_handler: DataHandler):
        self.spotify = spotify_client
        self.numeric_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy',
                             'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                             'popularity', 'speechiness', 'tempo']

        self.song_cluster_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=20))
        ])

        self.data = data_handler.data
        numeric_data = self.data[self.numeric_cols].select_dtypes(np.number)
        self.data['cluster_label'] = self.song_cluster_pipeline.fit_predict(numeric_data)

    def _get_song_data(self, song) -> np.ndarray | None:
        if song.get('name') is not None:
            try:
                return self.data[self.data['name'] == song['name']].iloc[0]
            except IndexError:
                return self.spotify.find_song_by_track(song['name'])

        if song.get('artist') is not None:
            try:
                return self.data[self.data['artists'].apply(lambda x: song['artist'].lower() in x.lower())]
            except IndexError:
                return self.spotify.find_song_by_artist(song['artist'])

        return None

    def _get_mean_vector(self, song_list) -> np.ndarray | None:
        song_vectors = []
        for song in song_list:
            song_data = self._get_song_data(song)

            if song_data is None or song_data.empty:
                return None

            try:
                if isinstance(song_data, pd.DataFrame):
                    song_vector = song_data[self.numeric_cols].mean().values
                else:
                    song_vector = song_data[self.numeric_cols].values
                song_vectors.append(song_vector)
            except KeyError:
                continue

        if not song_vectors:
            return None

        return np.mean(np.array(song_vectors), axis=0)

    def recommend_songs(self, song_list, n_songs=6) -> np.ndarray | None:
        metadata_cols = ['artists', 'name', 'year']
        song_center = self._get_mean_vector(song_list)

        if song_center is None:
            return None

        scaler = self.song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(self.data[self.numeric_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))

        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = np.argsort(distances)[:, :n_songs][0]

        rec_songs = self.data.iloc[index]
        df_rec_songs = rec_songs[metadata_cols].to_dict(orient='records')

        return pd.DataFrame(df_rec_songs).rename(columns={
            'artists': 'Artist/Band',
            'name': 'Track',
            'year': 'Year'
        })
