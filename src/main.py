import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clients import KaggleClient, SpotifyClient
from src.config import DATA_DIR
from src.data import DataHandler, SpotifyRecommendationSystem
from src.view import StreamlitInterface


def main():
    kaggle_client = KaggleClient()
    kaggle_client.download_spotify_dataset()

    spotify_client = SpotifyClient()
    data_handler = DataHandler(os.path.join(DATA_DIR, 'data.csv'))
    recommendation_system = SpotifyRecommendationSystem(spotify_client, data_handler)

    StreamlitInterface(recommendation_system).display_recommendations()


if __name__ == "__main__":
    main()
