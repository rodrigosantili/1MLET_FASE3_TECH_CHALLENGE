import datetime
import time
import streamlit as st
import numpy as np
from save_load_model import load_model_joblib
from sklearn.metrics import accuracy_score, classification_report
from evaluation_plots import (plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_lift_curve,
                              plot_learning_curve, plot_feature_importance, plot_probability_histogram, plot_kappa_statistic)
from eda_plots import (plot_target_distribution, plot_numerical_distributions, plot_correlation_matrix, plot_diameter_density,
                       calculate_vif, plot_feature_importance_eda, plot_scatter_miss_distance_velocity, plot_temporal_distribution)
from cross_validate import *


# Definir layout da página para ser mais largo - mover para o topo do arquivo
st.set_page_config(page_title="Previsão de Asteroides", layout="wide")

# Lista de modelos disponíveis
model_options = {
    "XGBoost": 'xgboost_model.joblib',
    "RandomForest": 'random_forest_model.joblib',
    "LogisticRegression": 'logistic_regression_model.joblib',
    "KNN": 'knn_model.joblib',
    "SVM": 'svm_model.joblib',
    "Perceptron": 'perceptron_model.joblib',
    "MLP": 'mlp_model.joblib'
}


# Função para carregar o modelo selecionado
def load_selected_model(model_name):
    model_filename = model_options[model_name]
    return load_model_joblib(model_filename)


# Caixa de seleção para o usuário escolher o modelo
selected_model_name = st.selectbox("Selecione o modelo para fazer a previsão:", options=model_options.keys())

# Carregar o modelo escolhido
model = load_selected_model(selected_model_name)
scaler = load_model_joblib('scaler.joblib')
pca = load_model_joblib('pca.joblib')
X_train = load_model_joblib('X_train.joblib')
X_test = load_model_joblib('X_test.joblib')
y_train = load_model_joblib('y_train.joblib')
y_test = load_model_joblib('y_test.joblib')

# Exibir uma mensagem com o modelo escolhido
st.write(f"Modelo {selected_model_name} carregado com sucesso.")

st.markdown("---")

# Processar os dados para gerar X_train, X_test, y_train, y_test
df = process_asteroid_data_from_json(folder_path='..', filename='../data/fetched_asteroids.json')


# Criar três colunas e usar a coluna do meio para centralizar a imagem
col1, col2, col3 = st.columns([1, 2, 1])

# Exibir a imagem na coluna do meio
with col1:
    st.text("")
    st.text("")
    st.image('image/asteroides.png', width=150)


# Função para converter ano em epoch
def convert_year_to_epoch(year):
    date_str = f"01-01-{year} 00:00:00"
    dt = datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")
    epoch_time = int(time.mktime(dt.timetuple()) * 1000)
    return epoch_time


def prediction_page():
    st.title("Previsão de Asteroides Potencialmente Perigosos")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    # Criar colunas para os campos de entrada
    col1, col2 = st.columns(2)

    with col1:
        # Magnitude Absoluta H
        st.write("**Magnitude Absoluta H**:")
        st.dialog("Refere-se ao brilho intrínseco do asteroide.")
        absolute_magnitude_h = st.number_input('Magnitude Absoluta H (min 9.25, max 30.1)', value=16.57)

        # Diâmetro Estimado Máximo
        st.write("**Diâmetro Estimado Máximo (km)**: ")
        st.dialog("Estimativa do maior diâmetro do asteroide.")
        estimated_diameter_max_km = st.number_input('Diâmetro Estimado Máximo (km) (min 0.0056, max 83.95)',
                                                    value=2.884297)

        # Velocidade Relativa
        st.write("**Velocidade Relativa (km/s)**:")
        st.dialog("Velocidade do asteroide em relação à Terra.")
        relative_velocity_kms = st.number_input('Velocidade Relativa (km/s) (min 0.075, max 94.57)', value=27.008071)

        # Distância de Passagem
        st.write("**Distância de Passagem (km)**:")
        st.dialog("Distância mínima da Terra durante a aproximação do asteroide.")
        miss_distance_km = st.number_input('Distância de Passagem (km) (min 12913.14, max 299134700)',
                                           value=12638584.486877)

        # Ano de Aproximação
        st.write("**Ano de Aproximação**:")
        st.dialog("Ano da aproximação mais próxima em formato epoch.")
        year = st.number_input('Ano de Aproximação (min 1900, max 2100)', value=2024, min_value=1900, max_value=2100)
        epoch_date_close_approach = convert_year_to_epoch(year)

        # Tipo de Classe de Órbita
        st.write("**Tipo de Classe de Órbita**:")
        st.dialog("Define a classe de órbita do asteroide. Valores: (0=Não classificado, 1=Amor, 2=Apollo, 3=Aten)")
        orbit_class_type = st.number_input('Tipo de Classe de Órbita (0, 1, 2, 3)', value=1.0)

    with col2:
        # Excentricidade
        st.write("**Excentricidade**:")
        st.dialog("Mede a forma da órbita (0 = circular, 1 = extremamente alongada).")
        eccentricity = st.number_input('Excentricidade (min 0.013, max 0.968)', value=0.826952)

        # Distância Periélio
        st.write("**Distância Periélio (AU)**:")
        st.dialog("Distância mínima do asteroide ao Sol em unidades astronômicas.")
        perihelion_distance = st.number_input('Distância Periélio (AU) (min 0.0704, max 1.299)', value=0.186553)

        # Inclinação
        st.write("**Inclinação (graus)**:")
        st.dialog("Inclinação da órbita do asteroide em relação ao plano da eclíptica.")
        inclination = st.number_input('Inclinação (graus) (min 0.054, max 154.35)', value=22.80351)

        # Mínima Interseção de Órbita
        st.write("**Mínima Interseção de Órbita (AU)**:")
        st.dialog("Distância mínima entre a órbita do asteroide e a da Terra.")
        minimum_orbit_intersection = st.number_input('Mínima Interseção de Órbita (min 0.000025, max 0.708)',
                                                     value=0.034215)

        # Período Orbital
        st.write("**Período Orbital (dias)**:")
        st.dialog("Tempo que o asteroide leva para completar uma órbita ao redor do Sol.")
        orbital_period = st.number_input('Período Orbital (dias) (min 176.57, max 27441.14)', value=408.837519)

        # Invariante de Tisserand de Júpiter
        st.write("**Invariante de Tisserand de Júpiter**:")
        st.dialog("Parâmetro dinâmico que mede a influência gravitacional de Júpiter no asteroide.")
        jupiter_tisserand_invariant = st.number_input('Invariante de Tisserand de Júpiter (min 1.316, max 9.039)',
                                                      value=5.299)

    # Prever com base nas entradas
    if st.button('Prever'):
        try:
            features = np.array(
                [[absolute_magnitude_h, estimated_diameter_max_km, relative_velocity_kms, miss_distance_km,
                  epoch_date_close_approach, orbit_class_type, eccentricity, perihelion_distance,
                  inclination, minimum_orbit_intersection, orbital_period, jupiter_tisserand_invariant]])
            scaled_features = scaler.transform(features)  # Aplicar scaler
            pca_features = pca.transform(scaled_features)  # Aplicar PCA
            prediction = model.predict(pca_features)[0]  # Fazer previsão com o modelo selecionado
            st.title(f"Previsão: {'Asteroide Perigoso' if prediction == 1 else 'Asteroide Não Perigoso'}")
        except Exception as e:
            st.error(f"Erro ao processar a previsão: {e}")


def model_evaluation_page():
    st.title("Avaliação do Modelo")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    y_pred = model.predict(X_test)

    # Verifica se o modelo possui o atributo 'predict_proba'
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None  # Para modelos como Perceptron

    if y_proba is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Matriz de Confusão")
            st.write(
                "**Descrição:** Mostra a relação entre as previsões do modelo e os valores reais, organizando os dados em Verdadeiros Positivos (TP), Verdadeiros Negativos (TN), Falsos Positivos (FP) e Falsos Negativos (FN).")
            st.write(
                "**Avaliação:** Idealmente, TP e TN devem ser altos e FP e FN baixos, indicando que o modelo acerta a maioria das previsões.")
            fig_confusion_matrix = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig_confusion_matrix)

        with col2:
            st.write("### Curva ROC (Receiver Operating Characteristic)")
            st.write(
                "**Descrição:** Mostra a taxa de verdadeiros positivos (sensibilidade) contra a taxa de falsos positivos em vários limiares de decisão.")
            st.write(
                "**Avaliação:** Quanto mais próxima a curva estiver do canto superior esquerdo, melhor. O AUC (Área sob a curva) próximo de 1 indica excelente desempenho.")
            fig_roc_curve = plot_roc_curve(y_test, y_proba)
            st.pyplot(fig_roc_curve)

        col3, col4 = st.columns(2)

        with col3:
            st.write("### Curva Precision-Recall")
            st.write(
                "**Descrição:** Relaciona a precisão (percentual de previsões corretas dentre as positivas) com o recall (sensibilidade) em diferentes limiares de decisão.")
            st.write(
                "**Avaliação:** Idealmente, a curva deve ser alta, indicando que o modelo equilibra bem precisão e recall.")
            fig_pr_curve = plot_precision_recall_curve(y_test, y_proba)
            st.pyplot(fig_pr_curve)

        with col4:
            st.write("### Curva Lift")
            st.write(
                "**Descrição:** Avalia a performance do modelo ao comparar a taxa de respostas observadas versus a taxa de respostas esperadas em uma população classificada pelo modelo.")
            st.write(
                "**Avaliação:** A curva deve mostrar um aumento significativo nos primeiros percentis, indicando que o modelo é eficaz para segmentar corretamente os exemplos positivos.")
            fig_lift_curve = plot_lift_curve(y_test, y_proba)
            st.pyplot(fig_lift_curve)

        col5, col6 = st.columns(2)

        with col5:
            st.write("### Curva de Aprendizado")
            st.write(
                "**Descrição:** Mostra a performance do modelo em termos de erro conforme o número de amostras aumenta, dividida em erro de treinamento e erro de validação.")
            st.write(
                "**Avaliação:** O modelo ideal apresenta uma redução no erro de validação à medida que mais dados são usados para treinamento, com a curva de erro de validação convergindo para a curva de erro de treinamento.")
            fig_learning_curve = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig_learning_curve)

        with col6:
            st.write("### Importância das Features")
            st.write("**Descrição:** Exibe a contribuição de cada feature no processo de tomada de decisão do modelo.")
            st.write(
                "**Avaliação:** Features com maior importância têm mais impacto nas previsões do modelo. Avalie se as features mais importantes fazem sentido para o problema em questão.")
            fig_feature_importance = plot_feature_importance(model, X_test, y_test)
            st.pyplot(fig_feature_importance)

        col7, col8 = st.columns(2)

        with col7:
            st.write("### Histograma de Probabilidades")
            st.write("**Descrição:** Mostra a distribuição das probabilidades previstas para a classe positiva (1).")
            st.write(
                "**Avaliação:** Um modelo bem calibrado deve ter uma distribuição clara, com alta probabilidade para a classe positiva e baixa para a negativa.")
            fig_probability_histogram = plot_probability_histogram(y_proba)
            st.pyplot(fig_probability_histogram)

        with col8:
            st.write("### Estatística de Kappa")
            st.write(
                "**Descrição:** Mede o grau de concordância entre as previsões do modelo e os valores reais, ajustando para concordâncias ao acaso.")
            st.write(
                "**Avaliação:** O valor de Kappa varia de -1 a 1, onde 1 indica perfeita concordância, 0 indica concordância ao acaso, e valores negativos indicam desempenho abaixo do esperado.")
            fig_kappa_statistic = plot_kappa_statistic(y_test, y_pred)
            st.pyplot(fig_kappa_statistic)
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Matriz de Confusão")
            st.write(
                "**Descrição:** Mostra a relação entre as previsões do modelo e os valores reais, organizando os dados em Verdadeiros Positivos (TP), Verdadeiros Negativos (TN), Falsos Positivos (FP) e Falsos Negativos (FN).")
            st.write(
                "**Avaliação:** Idealmente, TP e TN devem ser altos e FP e FN baixos, indicando que o modelo acerta a maioria das previsões.")
            fig_confusion_matrix = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig_confusion_matrix)

        with col2:
            st.write("### Curva de Aprendizado")
            st.write(
                "**Descrição:** Mostra a performance do modelo em termos de erro conforme o número de amostras aumenta, dividida em erro de treinamento e erro de validação.")
            st.write(
                "**Avaliação:** O modelo ideal apresenta uma redução no erro de validação à medida que mais dados são usados para treinamento, com a curva de erro de validação convergindo para a curva de erro de treinamento.")
            fig_learning_curve = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig_learning_curve)

        col3, col4 = st.columns(2)

        with col3:
            st.write("### Importância das Features")
            st.write("**Descrição:** Exibe a contribuição de cada feature no processo de tomada de decisão do modelo.")
            st.write(
                "**Avaliação:** Features com maior importância têm mais impacto nas previsões do modelo. Avalie se as features mais importantes fazem sentido para o problema em questão.")
            fig_feature_importance = plot_feature_importance(model, X_test, y_test)
            st.pyplot(fig_feature_importance)

        with col4:
            st.write("### Estatística de Kappa")
            st.write(
                "**Descrição:** Mede o grau de concordância entre as previsões do modelo e os valores reais, ajustando para concordâncias ao acaso.")
            st.write(
                "**Avaliação:** O valor de Kappa varia de -1 a 1, onde 1 indica perfeita concordância, 0 indica concordância ao acaso, e valores negativos indicam desempenho abaixo do esperado.")
            fig_kappa_statistic = plot_kappa_statistic(y_test, y_pred)
            st.pyplot(fig_kappa_statistic)


def eda_page():
    st.title("Análise Exploratória de Asteroides")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    col9, col10 = st.columns(2)

    with col9:
        st.write("### Distribuição de Asteroides Potencialmente Perigosos")
        st.write("**Descrição:** Este gráfico mostra a proporção de asteroides que são considerados potencialmente perigosos versus os que não são.")
        st.write("**Avaliação:** A análise ajuda a entender o equilíbrio das classes, o que é importante para avaliar se o modelo está lidando com classes desbalanceadas.")
        fig_target_dist = plot_target_distribution(df)
        st.pyplot(fig_target_dist)

    with col10:
        st.write("### Matriz de Correlação")
        st.write("**Descrição:** A matriz de correlação mostra a relação entre as variáveis numéricas do dataset. Cada célula contém o coeficiente de correlação entre dois atributos.")
        st.write("**Avaliação:** Correlações altas (positivas ou negativas) podem indicar redundância de features, o que pode ser problemático para alguns modelos.")
        fig_corr_matrix = plot_correlation_matrix(df)
        st.pyplot(fig_corr_matrix)

    col11, col12 = st.columns(2)

    with col11:
        st.write("### Distribuição dos Diâmetros por Periculosidade")
        st.write("**Descrição:** Este gráfico exibe a distribuição do diâmetro dos asteroides em relação à sua periculosidade.")
        st.write("**Avaliação:** A visualização ajuda a verificar se asteroides maiores têm maior probabilidade de serem considerados perigosos.")
        fig_diameter_density = plot_diameter_density(df)
        st.pyplot(fig_diameter_density)

    with col12:
        st.write("### Importância das Características")
        st.write("**Descrição:** Exibe a importância relativa das features para o modelo.")
        st.write("**Avaliação:** Identifica quais características mais influenciam a classificação dos asteroides como potencialmente perigosos ou não.")
        fig_feature_importance = plot_feature_importance_eda(df)
        st.pyplot(fig_feature_importance)

    col13, col14 = st.columns(2)

    with col13:
        st.write("### Dispersão entre Distância de Aproximação e Velocidade Relativa")
        st.write("**Descrição:** Gráfico de dispersão entre a distância de aproximação mínima e a velocidade relativa dos asteroides.")
        st.write("**Avaliação:** Permite verificar se há uma relação entre a distância de aproximação e a velocidade, ajudando a identificar padrões de comportamento entre asteroides mais próximos.")
        fig_scatter = plot_scatter_miss_distance_velocity(df)
        st.pyplot(fig_scatter)

    with col14:
        st.write("### Distribuição Temporal das Aproximações")
        st.write("**Descrição:** Mostra a distribuição das datas em que os asteroides fazem aproximações à Terra.")
        st.write("**Avaliação:** Ajuda a entender a periodicidade ou se há tendências temporais no comportamento dos asteroides.")
        fig_temporal_dist = plot_temporal_distribution(df)
        st.pyplot(fig_temporal_dist)

    col15, col16 = st.columns(2)

    with col15:
        st.write("### Distribuição das Variáveis Numéricas")
        st.write("**Descrição:** Exibe a distribuição das principais variáveis numéricas no dataset.")
        st.write("**Avaliação:** Permite observar a distribuição dos dados e identificar outliers, assim como assimetrias.")
        fig_num_distributions = plot_numerical_distributions(df)
        st.pyplot(fig_num_distributions)

    with col16:
        st.write("### Verificação de Multicolinearidade (VIF)")
        st.write("**Descrição:** O Variance Inflation Factor (VIF) mede a multicolinearidade entre as features.")
        st.write("**Avaliação:** Um VIF elevado (acima de 10) pode indicar problemas de multicolinearidade, sugerindo que algumas variáveis podem ser redundantes e devem ser removidas.")
        vif_data = calculate_vif(df)
        st.dataframe(vif_data)


def cross_validation():
    st.title(f"Validação Cruzada (K-Fold)")

    st.write(f"Realizando validação cruzada para o modelo: {selected_model_name}")

    # Realiza a validação cruzada com base no modelo selecionado
    if selected_model_name == "XGBoost":
        f1_score = cross_validate_xgboost_f1(X_train, y_train)
    elif selected_model_name == "RandomForest":
        f1_score = cross_validate_random_forest_f1(X_train, y_train)
    elif selected_model_name == "LogisticRegression":
        f1_score = cross_validate_logistic_regression_f1(X_train, y_train)
    elif selected_model_name == "KNN":
        f1_score = cross_validate_knn_f1(X_train, y_train)
    elif selected_model_name == "SVM":
        f1_score = cross_validate_svm_f1(X_train, y_train)
    elif selected_model_name == "Perceptron":
        f1_score = cross_validate_perceptron_f1(X_train, y_train)
    elif selected_model_name == "MLP":
        f1_score = cross_validate_mlp_f1(X_train, y_train)
    else:
        st.error("Modelo selecionado não encontrado.")
        return

    # Exibe o resultado da validação cruzada
    st.write(f"Modelo {selected_model_name} - F1 Score (ponderado) da Validação Cruzada: {f1_score:.4f}")

    st.title(f"Resultados de Previsão com Dados de Teste")

    # Fazer previsões com os dados de teste
    y_pred_rf = model.predict(X_test)

    # Calcular a acurácia e o classification report
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)

    # Exibir uma mensagem com o modelo escolhido
    st.write(f"{selected_model_name} - Accuracy: {accuracy_rf}")

    # Exibir uma mensagem com o modelo escolhido
    st.text(f"{selected_model_name} - Classification Report:\n{report_rf}")


def pca_analysis():
    st.write(f"### Variancia PCA")

    # Exibir a variância explicada
    explained_variance = pca.explained_variance_ratio_
    st.write(
        "**Descrição:** O gráfico de variância explicada pelos componentes principais mostra a quantidade de informação capturada por cada componente.")
    st.write(
        "**Avaliação:** A análise de PCA ajuda a identificar quais componentes principais contêm mais variância nos dados, facilitando a redução dimensional e visualização de características importantes no modelo. Os valores de variância explicada por componente variam de 0 a 1")
    st.text(f"Variância explicada por cada componente: {explained_variance}")
    st.text(f"Variância total explicada: {sum(explained_variance)}")


def main():
    st.sidebar.title("Menu de Navegação")
    page = st.sidebar.selectbox("Selecione a Página", ["Previsão", "Avaliação do Modelo", "Análise Exploratória EDA", "PCA", "Cross Validation"])

    if page == "Previsão":
        prediction_page()
    elif page == "Avaliação do Modelo":
        model_evaluation_page()
    elif page == "Análise Exploratória EDA":
        eda_page()
    elif page == "PCA":
        pca_analysis()
    elif page == "Cross Validation":
        cross_validation()



if __name__ == '__main__':
    main()
