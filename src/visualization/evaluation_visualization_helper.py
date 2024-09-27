import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score


class EvaluationVisualizationHelper:
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred) -> plt.Figure:
        """
        Plot the confusion matrix
        :param y_test: The true labels
        :param y_pred: The predicted labels
        :return: The confusion matrix plot
        """
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        return fig

    @staticmethod
    def plot_roc_curve(y_test, y_proba) -> plt.Figure:
        """
        Plot the ROC curve
        :param y_test: The true labels
        :param y_proba: The predicted probabilities
        :return: The ROC curve plot
        """
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        return fig

    @staticmethod
    def plot_precision_recall_curve(y_test, y_proba) -> plt.Figure:
        """
        Plot the Precision-Recall curve
        :param y_test: The true labels
        :param y_proba: The predicted probabilities
        :return: The Precision-Recall curve plot
        """
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, color='b', label='PR curve (AP = %0.2f)' % average_precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        return fig

    @staticmethod
    def plot_lift_curve(y_test, y_proba) -> plt.Figure:
        """
        Plot the Lift curve
        :param y_test: The true labels
        :param y_proba: The predicted probabilities
        :return: The Lift curve plot
        """
        sorted_y_test = np.array(y_test)[np.argsort(y_proba)[::-1]]
        cum_positive_rate = np.cumsum(sorted_y_test) / np.sum(sorted_y_test)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(np.arange(1, len(y_test) + 1) / len(y_test), cum_positive_rate, label='Lift curve')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('Percentile')
        plt.ylabel('Cumulative True Positive Rate')
        plt.title('Lift Curve')
        plt.legend(loc="upper left")
        return fig

    @staticmethod
    def plot_learning_curve(model, x_train, y_train) -> plt.Figure:
        """
        Plot the learning curve
        :param model: The trained model
        :param x_train: The training features
        :param y_train: The training labels
        :return: The learning curve plot
        """
        train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train,
                                                                cv=5, n_jobs=-1,
                                                                train_sizes=np.linspace(0.1, 1.0, 5))
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(train_sizes, train_scores_mean, label='Training')
        ax.plot(train_sizes, test_scores_mean, label='Validation')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend(loc="lower right")
        return fig

    @staticmethod
    def plot_feature_importance(model, x_test, y_test) -> plt.Figure:
        """
        Plot the feature importance using a Random Forest model
        :param model: The trained model
        :param x_test: The testing features
        :param y_test: The testing labels
        :return: The feature importance plot
        """
        result = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance = result.importances_mean
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(x=importance, y=[f'Feature {i + 1}' for i in range(len(importance))], ax=ax)
        plt.title('Feature Importance (Permutation)')
        return fig

    @staticmethod
    def plot_probability_histogram(y_proba) -> plt.Figure:
        """
        Plot the histogram of probabilities
        :param y_proba: The predicted probabilities
        :return: The histogram plot
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(y_proba, kde=True, ax=ax, color='blue')
        plt.title('Probability Histogram')
        plt.xlabel('Probability of Being Hazardous')
        return fig

    @staticmethod
    def plot_kappa_statistic(y_test, y_pred) -> plt.Figure:
        """
        Plot the Kappa statistic
        :param y_test: The true labels
        :param y_pred: The predicted labels
        :return: The Kappa statistic plot
        """
        kappa = cohen_kappa_score(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, f'Kappa: {kappa:.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=24, color='blue')
        plt.title('Kappa Statistic')
        plt.axis('off')
        return fig
