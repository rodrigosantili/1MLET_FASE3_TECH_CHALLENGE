import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import NasaApiClient, AsteroidDatasetHandler, AsteroidDataPreprocessor, AsteroidFeatureAndTargetSplitter
from src.ml import ModelPersistenceHandler, ModelTrainer


load_dotenv()
API_KEY_NASA = os.getenv('API_KEY_NASA')


def main():
    nasa_client = NasaApiClient(API_KEY_NASA)
    asteroid_dataset_handler = AsteroidDatasetHandler()
    model_persistence_handler = ModelPersistenceHandler()
    asteroid_data_preprocessor = AsteroidDataPreprocessor(model_persistence_handler)
    asteroid_feature_and_target_splitter = AsteroidFeatureAndTargetSplitter(model_persistence_handler)
    model_trainer = ModelTrainer(model_persistence_handler)

    # Check if the file fetched_asteroids.json already exists
    if not os.path.exists(NasaApiClient.DEFAULT_ASTEROIDS_DATA_FILEPATH):
        print("Asteroids data not found. Collecting data from NASA API...")
        nasa_client.fetch_asteroids_data(max_objects=3000)
    else:
        print("Asteroids data found. Skipping data collection from NASA API...")

    # Process the data from the JSON file
    df = asteroid_dataset_handler.load_and_handle_asteroid_dataset()

    # Preprocess the data with PCA
    pca_df, scaler, pca = asteroid_data_preprocessor.preprocess_asteroid_data(df)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = asteroid_feature_and_target_splitter.split(df, pca_df)

    model_trainer.train_and_save_all_models(x_train, x_test, y_train, y_test)

    print("\nFinished all ML models training and saving.")


if __name__ == '__main__':
    main()
