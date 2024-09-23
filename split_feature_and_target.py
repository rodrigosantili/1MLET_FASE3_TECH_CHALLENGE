from sklearn.model_selection import train_test_split
import joblib


def split_features_and_target(df, pca_df, save_path='models/'):
    # 1. Definir as features (X) e a variável alvo (y)
    X = pca_df  # As features após a redução de dimensionalidade com PCA
    y = df['is_potentially_hazardous_asteroid']  # Variável alvo (periculosidade do asteroide)

    # 2. Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Exibir o tamanho dos conjuntos
    print(f"Tamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de teste: {X_test.shape}")

    # 3. Salvar os conjuntos usando joblib
    joblib.dump(X_train, f'{save_path}X_train.joblib')
    joblib.dump(X_test, f'{save_path}X_test.joblib')
    joblib.dump(y_train, f'{save_path}y_train.joblib')
    joblib.dump(y_test, f'{save_path}y_test.joblib')

    # Retornar os conjuntos
    return X_train, X_test, y_train, y_test
