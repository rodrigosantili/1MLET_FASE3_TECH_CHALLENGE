import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.ml.model_persistence_handler import ModelPersistenceHandler


class AsteroidDataPreprocessor:
    def __init__(self, model_persistence_handler: ModelPersistenceHandler):
        self.model_persistence_handler = model_persistence_handler
        self.scaler = None
        self.pca = None

    def preprocess_asteroid_data(self, df) -> tuple[pd.DataFrame, StandardScaler, PCA]:
        """
        Preprocess the asteroid data by standardizing the numerical columns and reducing the dimensionality with PCA.
        :param df: DataFrame with the asteroid data
        :return: Tuple with the DataFrame containing the principal components,
                 the fitted StandardScaler, and the fitted PCA
        """
        # 1. Remove the columns 'estimated_diameter_min_km' and 'semi_major_axis'
        df = df.drop(['estimated_diameter_min_km', 'semi_major_axis'], axis=1)

        # 2. Select numerical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # 3. Standardize the data with StandardScaler
        self.scaler = StandardScaler()
        scaled_df = self.scaler.fit_transform(df[numerical_columns])

        # Save the fitted scaler to a .joblib file
        self.model_persistence_handler.save_model_joblib(self.scaler, 'scaler.joblib')
        print("Scaler saved as 'models/scaler.joblib'.")

        # 4. Dimensionality reduction with PCA, retaining 95% of the variance
        self.pca = PCA(n_components=0.95)
        principal_components = self.pca.fit_transform(scaled_df)

        # Save the fitted PCA to a .joblib file
        self.model_persistence_handler.save_model_joblib(self.pca, 'pca.joblib')
        print("PCA saved as 'models/pca.joblib'.")

        # Create DataFrame with the principal components
        pca_df = pd.DataFrame(data=principal_components)

        # Return the DataFrame with the principal components, the scaler, and the fitted PCA
        return pca_df, self.scaler, self.pca
