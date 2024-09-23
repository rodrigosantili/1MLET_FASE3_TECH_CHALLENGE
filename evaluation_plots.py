import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score

# Função para gerar a Matriz de Confusão
def plot_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    return fig  # Retorna o gráfico

# Função para gerar a Curva ROC
def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    return fig  # Retorna o gráfico

# Função para gerar a Curva Precision-Recall
def plot_precision_recall_curve(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color='b', label='PR curve (AP = %0.2f)' % average_precision)
    plt.xlabel('Revocação')
    plt.ylabel('Precisão')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    return fig

# Função para gerar a Curva Lift
def plot_lift_curve(y_test, y_proba):
    sorted_probas = np.sort(y_proba)[::-1]
    sorted_y_test = np.array(y_test)[np.argsort(y_proba)[::-1]]
    cum_positive_rate = np.cumsum(sorted_y_test) / np.sum(sorted_y_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.arange(1, len(y_test) + 1) / len(y_test), cum_positive_rate, label='Lift curve')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Percentil')
    plt.ylabel('Taxa Acumulada de Verdadeiros Positivos')
    plt.title('Curva Lift')
    plt.legend(loc="upper left")
    return fig

# Função para gerar a Curva de Aprendizado
def plot_learning_curve(model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(train_sizes, train_scores_mean, label='Treinamento')
    ax.plot(train_sizes, test_scores_mean, label='Validação')
    plt.xlabel('Número de amostras de treinamento')
    plt.ylabel('Acurácia')
    plt.title('Curva de Aprendizado')
    plt.legend(loc="lower right")
    return fig

# Função para gerar a Importância das Features com Permutation Importance
def plot_feature_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importance = result.importances_mean
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(x=importance, y=[f'Feature {i + 1}' for i in range(len(importance))], ax=ax)
    plt.title('Importância das Features (Permutação)')
    return fig

# Função para gerar Histogramas de Probabilidades
def plot_probability_histogram(y_proba):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(y_proba, kde=True, ax=ax, color='blue')
    plt.title('Histograma de Probabilidades')
    plt.xlabel('Probabilidade de ser Perigoso')
    return fig

# Função para gerar a Estatística de Kappa
def plot_kappa_statistic(y_test, y_pred):
    kappa = cohen_kappa_score(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.text(0.5, 0.5, f'Kappa: {kappa:.2f}', horizontalalignment='center', verticalalignment='center',
            fontsize=24, color='blue')
    plt.title('Estatística de Kappa')
    plt.axis('off')
    return fig
