from typing import Any

from src.data import AsteroidDatasetHandler
from src.ml import ModelPersistenceHandler, ModelEvaluator


class ModelController:
    def __init__(self):
        self.model_persistence_handler = ModelPersistenceHandler()
        self.model_options = {
            "XGBoost": 'xgboost_model.joblib',
            "RandomForest": 'random_forest_model.joblib',
            "LogisticRegression": 'logistic_regression_model.joblib',
            "KNN": 'knn_model.joblib',
            "SVM": 'svm_model.joblib',
            "Perceptron": 'perceptron_model.joblib',
            "MLP": 'mlp_model.joblib'
        }
        self.df = AsteroidDatasetHandler().load_and_handle_asteroid_dataset()

    def load_selected_model(self, model_name) -> Any:
        """
        Load the selected model from the .joblib file.
        :param model_name: The name of the selected model
        :return: The loaded model
        """
        model_filename = self.model_options[model_name]
        return self.model_persistence_handler.load_model_joblib(model_filename)

    def load_data(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        """
        Load the data from the .joblib files.
        :return: Tuple with the scaler, PCA, training and testing sets for the features and target variable
        """
        scaler = self.model_persistence_handler.load_model_joblib('scaler.joblib')
        pca = self.model_persistence_handler.load_model_joblib('pca.joblib')
        x_train = self.model_persistence_handler.load_model_joblib('X_train.joblib')
        x_test = self.model_persistence_handler.load_model_joblib('X_test.joblib')
        y_train = self.model_persistence_handler.load_model_joblib('y_train.joblib')
        y_test = self.model_persistence_handler.load_model_joblib('y_test.joblib')
        return scaler, pca, x_train, x_test, y_train, y_test

    def get_f1_score(self, model, x_train, y_train):
        if model == "XGBoost":
            return ModelEvaluator.cross_validate_xgboost_f1(x_train, y_train)
        elif model == "RandomForest":
            return ModelEvaluator.cross_validate_random_forest_f1(x_train, y_train)
        elif model == "LogisticRegression":
            return ModelEvaluator.cross_validate_logistic_regression_f1(x_train, y_train)
        elif model == "KNN":
            return ModelEvaluator.cross_validate_knn_f1(x_train, y_train)
        elif model == "SVM":
            return ModelEvaluator.cross_validate_svm_f1(x_train, y_train)
        elif model == "Perceptron":
            return ModelEvaluator.cross_validate_perceptron_f1(x_train, y_train)
        elif model == "MLP":
            return ModelEvaluator.cross_validate_mlp_f1(x_train, y_train)
        else:
            return None
