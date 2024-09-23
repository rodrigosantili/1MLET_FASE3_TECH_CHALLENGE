import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import warnings


# Ignorar UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Random Forest
def cross_validate_random_forest_f1(X, y):
    model_rf = RandomForestClassifier(random_state=42)
    scores_rf = cross_val_score(model_rf, X, y, cv=5, scoring='f1_weighted')
    return scores_rf.mean()

# 2. Regressão Logística
def cross_validate_logistic_regression_f1(X, y):
    model_lr = LogisticRegression(random_state=42)
    scores_lr = cross_val_score(model_lr, X, y, cv=5, scoring='f1_weighted')
    return scores_lr.mean()

# 3. KNN
def cross_validate_knn_f1(X, y, n_neighbors=5):
    model_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores_knn = cross_val_score(model_knn, X, y, cv=5, scoring='f1_weighted')
    return scores_knn.mean()

# 4. SVM
def cross_validate_svm_f1(X, y):
    model_svm = SVC(random_state=42)
    scores_svm = cross_val_score(model_svm, X, y, cv=5, scoring='f1_weighted')
    return scores_svm.mean()

# 5. Perceptron (RNA Simples)
def cross_validate_perceptron_f1(X, y):
    model_perceptron = Perceptron(random_state=42, class_weight='balanced', max_iter=1000, tol=1e-3)
    scores_perceptron = cross_val_score(model_perceptron, X, y, cv=5, scoring='f1_weighted')
    return scores_perceptron.mean()

# 6. Multi-Layer Perceptron (MLP)
def cross_validate_mlp_f1(X, y):
    model_mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=0.001)
    scores_mlp = cross_val_score(model_mlp, X, y, cv=5, scoring='f1_weighted')
    return scores_mlp.mean()

# 7. XGBoost
def cross_validate_xgboost_f1(X, y):
    model_xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    scores_xgb = cross_val_score(model_xgb, X, y, cv=5, scoring='f1_weighted')
    return scores_xgb.mean()

