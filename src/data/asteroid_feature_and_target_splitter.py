from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class AsteroidFeatureAndTargetSplitter:
    def __init__(self, model_persistence_handler):
        self.model_persistence_handler = model_persistence_handler

    def split(self, df, pca_df) -> tuple[list, list, list, list]:
        """
        Split the features and target variable.
        :param df: The dataframe with the asteroid data
        :param pca_df: The dataframe with the principal components
        :return: Tuple with the training and testing sets for the features and target variable
        """
        # 1. Define the features (X) and the target variable (y)
        x = pca_df  # Features after dimensionality reduction with PCA
        y = df['is_potentially_hazardous_asteroid']  # Target variable (asteroid hazard)

        # 2. Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Display the size of the sets
        print(f"Training set size: {x_train.shape}")
        print(f"Testing set size: {x_test.shape}")

        # 4. Instanciando o objeto SMOTE
        smote = SMOTE(random_state=42)

        # 5. Aplicando o SMOTE apenas no conjunto de treinamento
        x_train, y_train = smote.fit_resample(x_train, y_train)

        # Display the size of the sets
        print(f"Training smote set size: {x_train.shape}")
        print(f"Testing smote set size: {x_test.shape}")

        # 6. Save the sets using joblib
        self.model_persistence_handler.save_model_joblib(x_train, 'x_train.joblib')
        self.model_persistence_handler.save_model_joblib(x_test, 'x_test.joblib')
        self.model_persistence_handler.save_model_joblib(y_train, 'y_train.joblib')
        self.model_persistence_handler.save_model_joblib(y_test, 'y_test.joblib')

        # Return the sets
        return x_train, x_test, y_train, y_test
