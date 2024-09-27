import os.path
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import MODELS_DIR
from .model_persistence_handler import ModelPersistenceHandler

# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


class ModelTrainer:

    def __init__(self, model_persistence_handler: ModelPersistenceHandler):
        self.model_persistence_handler = model_persistence_handler

    @staticmethod
    def train_random_forest(x_train, x_test, y_train, y_test) -> RandomForestClassifier:
        """
        Train a Random Forest model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: RandomForestClassifier: Trained Random Forest model
        """
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(x_train, y_train)
        y_pred_rf = model_rf.predict(x_test)

        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf)

        print(f"Random Forest - Accuracy: {accuracy_rf}")
        print(f"Random Forest - Classification Report:\n{report_rf}")

        return model_rf

    @staticmethod
    def train_logistic_regression(x_train, x_test, y_train, y_test) -> LogisticRegression:
        """
        Train a Logistic Regression model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: LogisticRegression: Trained Logistic Regression model
        """
        model_lr = LogisticRegression(random_state=42)
        model_lr.fit(x_train, y_train)
        y_pred_lr = model_lr.predict(x_test)

        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        report_lr = classification_report(y_test, y_pred_lr)

        print(f"Logistic Regression - Accuracy: {accuracy_lr}")
        print(f"Logistic Regression - Classification Report:\n{report_lr}")

        return model_lr

    @staticmethod
    def train_knn(x_train, x_test, y_train, y_test, n_neighbors=5) -> KNeighborsClassifier:
        """
        Train a K-Nearest Neighbors model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :param n_neighbors: (int): Number of neighbors to consider
        :return: KNeighborsClassifier: Trained K-Nearest Neighbors model
        """
        model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        model_knn.fit(x_train, y_train)
        y_pred_knn = model_knn.predict(x_test)

        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        report_knn = classification_report(y_test, y_pred_knn)

        print(f"KNN - Accuracy: {accuracy_knn}")
        print(f"KNN - Classification Report:\n{report_knn}")

        return model_knn

    @staticmethod
    def train_svm(x_train, x_test, y_train, y_test) -> SVC:
        """
        Train a Support Vector Machine model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: SVC: Trained Support Vector Machine model
        """
        model_svm = SVC(random_state=42, probability=True)
        model_svm.fit(x_train, y_train)
        y_pred_svm = model_svm.predict(x_test)

        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm)

        print(f"SVM - Accuracy: {accuracy_svm}")
        print(f"SVM - Classification Report:\n{report_svm}")

        return model_svm

    @staticmethod
    def train_perceptron(x_train, x_test, y_train, y_test) -> Perceptron:
        """
        Train a Perceptron model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: Perceptron: Trained Perceptron model
        """
        model_perceptron = Perceptron(random_state=42, class_weight='balanced', max_iter=1000, tol=1e-3)
        model_perceptron.fit(x_train, y_train)
        y_pred_perceptron = model_perceptron.predict(x_test)

        accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
        report_perceptron = classification_report(y_test, y_pred_perceptron)

        print(f"Perceptron - Accuracy: {accuracy_perceptron}")
        print(f"Perceptron - Classification Report:\n{report_perceptron}")

        return model_perceptron

    @staticmethod
    def train_mlp(x_train, x_test, y_train, y_test) -> MLPClassifier:
        """
        Train a Multi-Layer Perceptron model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: MLPClassifier: Trained Multi-Layer Perceptron model
        """
        model_mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=0.001)
        model_mlp.fit(x_train, y_train)
        y_pred_mlp = model_mlp.predict(x_test)

        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        report_mlp = classification_report(y_test, y_pred_mlp)

        print(f"MLP - Accuracy: {accuracy_mlp}")
        print(f"MLP - Classification Report:\n{report_mlp}")

        return model_mlp

    @staticmethod
    def train_xgboost(x_train, x_test, y_train, y_test) -> XGBClassifier:
        """
        Train an XGBoost model and evaluate its performance
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        :return: XGBClassifier: Trained XGBoost model
        """
        model_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        model_xgb.fit(x_train, y_train)
        y_pred_xgb = model_xgb.predict(x_test)

        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        report_xgb = classification_report(y_test, y_pred_xgb)

        print(f"XGBoost - Accuracy: {accuracy_xgb}")
        print(f"XGBoost - Classification Report:\n{report_xgb}")

        return model_xgb

    def train_and_save_all_models(self, x_train, x_test, y_train, y_test) -> None:
        """
        Train and save all models
        :param x_train: (DataFrame): Training features
        :param x_test: (DataFrame): Testing features
        :param y_train: (Series): Training labels
        :param y_test: (Series): Testing labels
        """
        models = {
            "random_forest": ModelTrainer.train_random_forest,
            "logistic_regression": ModelTrainer.train_logistic_regression,
            "knn": ModelTrainer.train_knn,
            "svm": ModelTrainer.train_svm,
            "perceptron": ModelTrainer.train_perceptron,
            "mlp": ModelTrainer.train_mlp,
            "xgboost": ModelTrainer.train_xgboost
        }

        for model_name, train_function in models.items():
            print(f"\nTraining model {model_name}...")
            model = train_function(x_train, x_test, y_train, y_test)
            self.model_persistence_handler.save_model_joblib(model, f'{model_name}_model.joblib')
