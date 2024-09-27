import requests
import json
import os

from src.config import DATA_DIR


class NasaApiClient:

    DEFAULT_MAX_OBJECTS = 1000
    DEFAULT_ASTEROIDS_DATA_FILEPATH = os.path.join(DATA_DIR, 'asteroids_data.json')

    def __init__(self, api_key, base_url="https://api.nasa.gov/neo/rest/v1/neo/browse"):
        self.api_key = api_key
        self.base_url = base_url

    def fetch_asteroids_data(self, max_objects=DEFAULT_MAX_OBJECTS) -> None:
        """
        Fetches data from the NASA API and saves it to a JSON file.
        :param max_objects: Maximum number of objects to fetch
        """
        fetched_objects = []
        page = 0

        while len(fetched_objects) < max_objects:
            url = f"{self.base_url}?page={page}&api_key={self.api_key}"
            response = self._make_api_request(url)

            if response is None:
                break

            data = response.json()
            if 'near_earth_objects' not in data:
                break

            fetched_objects.extend(data['near_earth_objects'][:max_objects - len(fetched_objects)])
            page += 1

        self._save_to_json(fetched_objects)

    def _make_api_request(self, url) -> requests.Response | None:
        """
        Makes a request to the NASA API.
        :param url: URL to make the request
        :return: Response object if successful, None otherwise
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
            print(f"Error accessing the NASA API: {response.status_code}")
        except Exception as e:
            print(f"Error during request: {e}")
        return None

    def _save_to_json(self, data) -> None:
        """
        Saves the fetched data to a JSON file.
        :param data: Data to save
        :param filename: Name of the file to save
        """
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        with open(self.DEFAULT_ASTEROIDS_DATA_FILEPATH, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"File saved in: {self.DEFAULT_ASTEROIDS_DATA_FILEPATH}")
