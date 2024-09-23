import os
from kaggle import KaggleApi
from src.config import DATA_DIR, RES_DIR


class KaggleClient:
    def __init__(self, dataset_url='vatsalmavani/spotify-dataset'):
        self.dataset_url = dataset_url

    def download_spotify_dataset(self) -> None:
        expected_files = ['data_w_genres.csv', 'data_by_year.csv', 'data_by_genres.csv',
                          'data_by_artist.csv', 'data.csv']

        if all([os.path.exists(os.path.join(DATA_DIR, file)) for file in expected_files]):
            print(f"Dataset already downloaded.")
        else:
            api = KaggleApi()
            api.authenticate()
            print(f"Downloading dataset...")
            api.dataset_download_files(self.dataset_url, path=RES_DIR, unzip=True)
            print(f"Dataset downloaded successfully")
