import json
from datetime import datetime

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .nasa_api_client import NasaApiClient


class AsteroidDatasetHandler:
    def __init__(self, dataset=NasaApiClient.DEFAULT_ASTEROIDS_DATA_FILEPATH):
        self.dataset = dataset

    def load_and_handle_asteroid_dataset(self) -> pd.DataFrame | None:
        """
        Loads the asteroid data from the JSON file and processes it into a DataFrame.
        :return: DataFrame with the processed asteroid data
        """
        asteroids = self._load_json()

        if asteroids is None or not isinstance(asteroids, list):
            print(f"Error: Content of JSON file is not a list. Verify the file structure.")
            return None

        asteroid_data = self._extract_features(asteroids)
        df = pd.DataFrame(asteroid_data)
        self._convert_df_numeric_columns(df)
        self._encode_orbit_class_type(df)

        return df

    def _load_json(self) -> list | None:
        """
        Loads the JSON file containing the asteroid data.
        :return: List with the asteroid data
        """
        try:
            with open(self.dataset, 'r', encoding='utf-8') as file:
                print(f"Dataset {self.dataset} succesfully loaded!")
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {self.dataset} not found.")
            return None
        except ValueError as e:
            print(f"Error loading JSON: {e}")
            return None

    def _extract_features(self, asteroids) -> list:
        """
        Extracts the relevant features from the asteroid data.
        :param asteroids: List of NEO objects
        :return: List of dictionaries with the extracted features
        """
        current_epoch = datetime.now().timestamp() * 1000
        asteroid_data = []

        for neo in asteroids:
            if not isinstance(neo, dict):
                print(f"Error: NEO object not a dictionary. Verify data.")
                continue

            for approach in neo.get('close_approach_data', []):
                epoch_date_close_approach = approach.get('epoch_date_close_approach')
                if epoch_date_close_approach and epoch_date_close_approach <= current_epoch:
                    features = {
                        'absolute_magnitude_h': neo.get('absolute_magnitude_h'),
                        'estimated_diameter_min_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_min'),
                        'estimated_diameter_max_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max'),
                        'relative_velocity_kms': approach.get('relative_velocity', {}).get('kilometers_per_second'),
                        'miss_distance_km': approach.get('miss_distance', {}).get('kilometers'),
                        'epoch_date_close_approach': epoch_date_close_approach,
                        'orbit_class_type': neo.get('orbital_data', {}).get('orbit_class', {}).get('orbit_class_type'),
                        'eccentricity': neo.get('orbital_data', {}).get('eccentricity'),
                        'semi_major_axis': neo.get('orbital_data', {}).get('semi_major_axis'),
                        'perihelion_distance': neo.get('orbital_data', {}).get('perihelion_distance'),
                        'inclination': neo.get('orbital_data', {}).get('inclination'),
                        'minimum_orbit_intersection': neo.get('orbital_data', {}).get('minimum_orbit_intersection'),
                        'orbital_period': neo.get('orbital_data', {}).get('orbital_period'),
                        'jupiter_tisserand_invariant': neo.get('orbital_data', {}).get('jupiter_tisserand_invariant'),
                        'is_potentially_hazardous_asteroid': neo.get('is_potentially_hazardous_asteroid')
                    }
                    asteroid_data.append(features)
        return asteroid_data

    def _convert_df_numeric_columns(self, df) -> None:
        """
        Converts the numeric columns in the DataFrame to float64.
        :param df: DataFrame to convert
        """
        numeric_columns = [
            'relative_velocity_kms',
            'miss_distance_km',
            'eccentricity',
            'semi_major_axis',
            'perihelion_distance',
            'inclination',
            'minimum_orbit_intersection',
            'orbital_period',
            'jupiter_tisserand_invariant'
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
            else:
                print(f"Column '{column}' not found within the DataFrame.")

    def _encode_orbit_class_type(self, df) -> None:
        """
        Encodes the 'orbit_class_type' column using LabelEncoder.
        :param df: DataFrame to encode
        """
        if 'orbit_class_type' in df.columns:
            df['orbit_class_type'] = LabelEncoder().fit_transform(df['orbit_class_type'])
        else:
            print("Column 'orbit_class_type' not found within the DataFrame.")
