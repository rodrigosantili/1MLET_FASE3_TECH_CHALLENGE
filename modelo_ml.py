import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings


# Ignorar UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


# 1. Random Forest
def train_random_forest(X_train, X_test, y_train, y_test):
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    print(f"Random Forest - Accuracy: {accuracy_rf}")
    print(f"Random Forest - Classification Report:\n{report_rf}")

    return model_rf


# 2. Regressão Logística
def train_logistic_regression(X_train, X_test, y_train, y_test):
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)

    print(f"Logistic Regression - Accuracy: {accuracy_lr}")
    print(f"Logistic Regression - Classification Report:\n{report_lr}")

    return model_lr


# 3. KNN
def train_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn)

    print(f"KNN - Accuracy: {accuracy_knn}")
    print(f"KNN - Classification Report:\n{report_knn}")

    return model_knn


# 4. SVM
def train_svm(X_train, X_test, y_train, y_test):
    model_svm = SVC(random_state=42, probability=True)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    report_svm = classification_report(y_test, y_pred_svm)

    print(f"SVM - Accuracy: {accuracy_svm}")
    print(f"SVM - Classification Report:\n{report_svm}")

    return model_svm


# 5. Perceptron (RNA Simples)
def train_perceptron(X_train, X_test, y_train, y_test):
    model_perceptron = Perceptron(random_state=42, class_weight='balanced', max_iter=1000, tol=1e-3)
    model_perceptron.fit(X_train, y_train)
    y_pred_perceptron = model_perceptron.predict(X_test)

    accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
    report_perceptron = classification_report(y_test, y_pred_perceptron)

    print(f"Perceptron - Accuracy: {accuracy_perceptron}")
    print(f"Perceptron - Classification Report:\n{report_perceptron}")

    return model_perceptron


# 6. Multi-Layer Perceptron (MLP)
def train_mlp(X_train, X_test, y_train, y_test):
    model_mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=0.001)
    model_mlp.fit(X_train, y_train)
    y_pred_mlp = model_mlp.predict(X_test)

    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    report_mlp = classification_report(y_test, y_pred_mlp)

    print(f"MLP - Accuracy: {accuracy_mlp}")
    print(f"MLP - Classification Report:\n{report_mlp}")

    return model_mlp

# 7. XGBoost
def train_xgboost(X_train, X_test, y_train, y_test):
    model_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)

    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb)

    print(f"XGBoost - Accuracy: {accuracy_xgb}")
    print(f"XGBoost - Classification Report:\n{report_xgb}")

    return model_xgb
