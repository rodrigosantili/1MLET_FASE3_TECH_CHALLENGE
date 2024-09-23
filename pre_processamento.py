import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from save_load_model import save_model_joblib


def preprocess_asteroid_data(df):
    # 1. Remover as colunas 'estimated_diameter_min_km' e 'semi_major_axis'
    df = df.drop(['estimated_diameter_min_km', 'semi_major_axis'], axis=1)
    # print("Colunas restantes após a remoção:", df.columns)

    # 2. Selecionar colunas numéricas
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # 3. Padronização dos dados com StandardScaler
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[numerical_columns])

    # Salvar o scaler ajustado em um arquivo .joblib
    save_model_joblib(scaler, 'scaler.joblib')
    print("Scaler salvo como 'models/scaler.joblib'.")

    # 4. Redução de dimensionalidade com PCA, mantendo 95% da variação
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(scaled_df)

    # Salvar o PCA ajustado em um arquivo .joblib
    save_model_joblib(pca, 'pca.joblib')
    print("PCA salvo como 'models/pca.joblib'.")

    # Criar DataFrame com as componentes principais
    pca_df = pd.DataFrame(data=principal_components)

    # Retornar o DataFrame com as componentes principais, o scaler e o PCA ajustado
    return pca_df, scaler, pca
