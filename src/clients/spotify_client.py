import os

import pandas as pd
from dotenv import load_dotenv
from pandas import DataFrame
from spotipy import Spotify, SpotifyClientCredentials


class SpotifyClient:
    def __init__(self):
        load_dotenv()
        client_id, client_secret = os.environ["SPOTIFY_CLIENT_ID"], os.environ["SPOTIFY_CLIENT_SECRET"]
        auth_manager = SpotifyClientCredentials(client_id, client_secret)
        self.conn = Spotify(auth_manager)

    def _find_song(self, query) -> DataFrame | None:
        results = self.conn.search(q=query, limit=1)
        if not results['tracks']['items']:
            return None

        track = results['tracks']['items'][0]
        track_id = track['id']
        audio_features = self.conn.audio_features(track_id)[0]

        song_data = {
            'name': track['name'],
            'artists': track['artists'][0]['name'],
            'explicit': int(track['explicit']),
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'year': int(track['album']['release_date'][:4]),
            **audio_features
        }

        return pd.DataFrame([song_data])

    def find_song_by_track(self, track_name) -> DataFrame | None:
        query = f'track:{track_name}'
        return self._find_song(query)

    def find_song_by_artist(self, artist_name) -> DataFrame | None:
        query = f'artist:{artist_name}'
        return self._find_song(query)
