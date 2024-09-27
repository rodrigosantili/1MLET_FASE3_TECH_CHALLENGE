import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)


class ModelEvaluator:
    @staticmethod
    def cross_validate_random_forest_f1(x, y) -> float:
        """
        Perform cross-validation using a Random Forest classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_rf = RandomForestClassifier(random_state=42)
        scores_rf = cross_val_score(model_rf, x, y, cv=5, scoring='f1_weighted')
        return scores_rf.mean()

    @staticmethod
    def cross_validate_logistic_regression_f1(x, y) -> float:
        """
        Perform cross-validation using a Logistic Regression classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_lr = LogisticRegression(random_state=42)
        scores_lr = cross_val_score(model_lr, x, y, cv=5, scoring='f1_weighted')
        return scores_lr.mean()

    @staticmethod
    def cross_validate_knn_f1(x, y, n_neighbors=5) -> float:
        """
        Perform cross-validation using a K-Nearest Neighbors classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :param n_neighbors: Number of neighbors to use for KNN
        :return: Mean F1 score from cross-validation
        """
        model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores_knn = cross_val_score(model_knn, x, y, cv=5, scoring='f1_weighted')
        return scores_knn.mean()

    @staticmethod
    def cross_validate_svm_f1(x, y) -> float:
        """
        Perform cross-validation using a Support Vector Machine classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_svm = SVC(random_state=42)
        scores_svm = cross_val_score(model_svm, x, y, cv=5, scoring='f1_weighted')
        return scores_svm.mean()

    @staticmethod
    def cross_validate_perceptron_f1(x, y) -> float:
        """
        Perform cross-validation using a Perceptron classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_perceptron = Perceptron(random_state=42, class_weight='balanced', max_iter=1000, tol=1e-3)
        scores_perceptron = cross_val_score(model_perceptron, x, y, cv=5, scoring='f1_weighted')
        return scores_perceptron.mean()

    @staticmethod
    def cross_validate_mlp_f1(x, y) -> float:
        """
        Perform cross-validation using a Multi-Layer Perceptron classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=0.001)
        scores_mlp = cross_val_score(model_mlp, x, y, cv=5, scoring='f1_weighted')
        return scores_mlp.mean()

    @staticmethod
    def cross_validate_xgboost_f1(x, y) -> float:
        """
        Perform cross-validation using an XGBoost classifier and calculate the mean F1 score

        :param x: Features for training the model
        :param y: Target labels for training the model
        :return: Mean F1 score from cross-validation
        """
        model_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        scores_xgb = cross_val_score(model_xgb, x, y, cv=5, scoring='f1_weighted')
        return scores_xgb.mean()
