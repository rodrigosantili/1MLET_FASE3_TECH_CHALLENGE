import os
from dotenv import load_dotenv
from fetch_data import fetch_asteroids, save_to_json
from tratamento import process_asteroid_data_from_json
from pre_processamento import preprocess_asteroid_data
from split_feature_and_target import split_features_and_target
from modelo_ml import train_random_forest, train_logistic_regression, train_knn, train_svm, train_perceptron, train_mlp, train_xgboost
from save_load_model import save_model_joblib

# Carregar a chave da API a partir do arquivo .env
load_dotenv()
API_KEY_NASA = os.getenv('API_KEY_NASA')

# Verificar se o arquivo fetched_asteroids.json já existe
if not os.path.exists('data/fetched_asteroids.json'):
    print("Arquivo 'fetched_asteroids.json' não encontrado. Coletando dados da API...")
    # Coletar os dados dos asteroides
    fetched_objects = fetch_asteroids(API_KEY_NASA, max_objects=5000)
    # Salvar os dados coletados em um arquivo JSON
    save_to_json(fetched_objects, 'data/fetched_asteroids.json')
else:
    print("Arquivo 'fetched_asteroids.json' encontrado. Pulando etapa de coleta de dados.")


# Processar os dados do arquivo JSON
df = process_asteroid_data_from_json(folder_path='.', filename='data/fetched_asteroids.json')

# Pré-processar os dados com PCA
pca_df, scaler, pca = preprocess_asteroid_data(df)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = split_features_and_target(df, pca_df)

# Dicionário de modelos para facilitar o loop
models = {
    "random_forest": train_random_forest,
    "logistic_regression": train_logistic_regression,
    "knn": train_knn,
    "svm": train_svm,
    "perceptron": train_perceptron,
    "mlp": train_mlp,
    "xgboost": train_xgboost
}

# Loop para treinar todos os modelos e salvar
for model_name, train_function in models.items():
    print(f"\nTreinando modelo {model_name}...")
    model = train_function(X_train, X_test, y_train, y_test)
    model_filename = f'{model_name}_model.joblib'
    save_model_joblib(model, model_filename)
    print(f"Modelo {model_name} salvo como {model_filename}")

print("\nTreinamento e salvamento de todos os modelos concluídos.")
