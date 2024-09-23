import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 1. Função para mostrar a distribuição da variável alvo
def plot_target_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='is_potentially_hazardous_asteroid', data=df, hue='is_potentially_hazardous_asteroid', legend=False)
    plt.title('Distribuição de Asteroides Potencialmente Perigosos')
    plt.xlabel('Asteróide Potencialmente Perigoso')
    plt.ylabel('Quantidade')
    return plt

# 2. Função para mostrar os histogramas das variáveis numéricas
def plot_numerical_distributions(df):
    df.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    return plt

# 3. Função para gerar a matriz de correlação
def plot_correlation_matrix(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_df = df[numerical_columns]
    plt.figure(figsize=(10, 6))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    return plt

# 4. Função para plotar a densidade dos diâmetros por periculosidade
def plot_diameter_density(df):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[df['is_potentially_hazardous_asteroid']]['estimated_diameter_min_km'],
                label='Perigoso - Min', fill=True, color='red')
    sns.kdeplot(df[df['is_potentially_hazardous_asteroid']]['estimated_diameter_max_km'],
                label='Perigoso - Max', fill=True, color='blue')
    plt.title('Distribuição dos Diâmetros Mínimo e Máximo por Periculosidade')
    plt.xlabel('Diâmetro em Quilômetros')
    plt.ylabel('Densidade')
    plt.legend()
    return plt

# 5. Função para calcular e exibir o VIF (Variance Inflation Factor)
def calculate_vif(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_df = df[numerical_columns].copy()
    numerical_df = sm.add_constant(numerical_df)
    vif_data = pd.DataFrame()
    vif_data['feature'] = numerical_df.columns
    vif_data['VIF'] = [variance_inflation_factor(numerical_df.values, i) for i in range(numerical_df.shape[1])]
    return vif_data

# 6. Função para mostrar a importância das features usando RandomForest
# Função para mostrar a importância das features usando RandomForest (EDA)
def plot_feature_importance_eda(df):
    X = df.drop('is_potentially_hazardous_asteroid', axis=1)
    y = df['is_potentially_hazardous_asteroid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='dodgerblue')
    plt.xlabel('Importância')
    plt.title('Importância das Características (Feature Importance - EDA)')
    plt.gca().invert_yaxis()
    return plt


# 7. Função para plotar a dispersão entre miss_distance_km e relative_velocity_kms
def plot_scatter_miss_distance_velocity(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='miss_distance_km', y='relative_velocity_kms', hue='is_potentially_hazardous_asteroid', data=df)
    plt.title('Dispersão entre Distância de Aproximação e Velocidade Relativa')
    return plt


# 8. Função para plotar a distribuição temporal das aproximações
def plot_temporal_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['epoch_date_close_approach'], bins=100)
    plt.title('Distribuição Temporal das Aproximações de Asteroides')
    plt.xlabel('Data')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    return plt
