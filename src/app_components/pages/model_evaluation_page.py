import streamlit as st

from src.app_components.controllers import ModelController
from src.visualization import EvaluationVisualizationHelper


class ModelEvaluationPage:
    def __init__(self,
                 model_controller: ModelController,
                 evaluation_visualization_helper: EvaluationVisualizationHelper):
        self.model_controller = model_controller
        self.evaluation_visualization_helper = evaluation_visualization_helper

    def display(self):
        # Criar três colunas e usar a coluna do meio para centralizar a imagem
        col1, col2, col3 = st.columns([1, 2, 1])

        # Exibir a imagem na coluna do meio
        with col1:
            st.text("")
            st.text("")
            st.image('image/asteroides.png', width=150)

        st.title("Avaliação do Modelo")
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        _, _, x_train, x_test, y_train, y_test = self.model_controller.load_data()
        model = self.model_controller.load_selected_model(st.session_state.selected_model_name)
        y_pred = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
        else:
            y_proba = None

        if y_proba is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Matriz de Confusão")
                st.write(
                    "**Descrição:** Mostra a relação entre as previsões do modelo e os valores reais, organizando os dados em Verdadeiros Positivos (TP), Verdadeiros Negativos (TN), Falsos Positivos (FP) e Falsos Negativos (FN).")
                st.write(
                    "**Avaliação:** Idealmente, TP e TN devem ser altos e FP e FN baixos, indicando que o modelo acerta a maioria das previsões.")
                fig_confusion_matrix = self.evaluation_visualization_helper.plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig_confusion_matrix)

            with col2:
                st.write("### Curva ROC (Receiver Operating Characteristic)")
                st.write(
                    "**Descrição:** Mostra a taxa de verdadeiros positivos (sensibilidade) contra a taxa de falsos positivos em vários limiares de decisão.")
                st.write(
                    "**Avaliação:** Quanto mais próxima a curva estiver do canto superior esquerdo, melhor. O AUC (Área sob a curva) próximo de 1 indica excelente desempenho.")
                fig_roc_curve = self.evaluation_visualization_helper.plot_roc_curve(y_test, y_proba)
                st.pyplot(fig_roc_curve)

            col3, col4 = st.columns(2)

            with col3:
                st.write("### Curva Precision-Recall")
                st.write(
                    "**Descrição:** Relaciona a precisão (percentual de previsões corretas dentre as positivas) com o recall (sensibilidade) em diferentes limiares de decisão.")
                st.write(
                    "**Avaliação:** Idealmente, a curva deve ser alta, indicando que o modelo equilibra bem precisão e recall.")
                fig_pr_curve = self.evaluation_visualization_helper.plot_precision_recall_curve(y_test, y_proba)
                st.pyplot(fig_pr_curve)

            with col4:
                st.write("### Curva Lift")
                st.write(
                    "**Descrição:** Avalia a performance do modelo ao comparar a taxa de respostas observadas versus a taxa de respostas esperadas em uma população classificada pelo modelo.")
                st.write(
                    "**Avaliação:** A curva deve mostrar um aumento significativo nos primeiros percentis, indicando que o modelo é eficaz para segmentar corretamente os exemplos positivos.")
                fig_lift_curve = self.evaluation_visualization_helper.plot_lift_curve(y_test, y_proba)
                st.pyplot(fig_lift_curve)

            col5, col6 = st.columns(2)

            with col5:
                st.write("### Curva de Aprendizado")
                st.write(
                    "**Descrição:** Mostra a performance do modelo em termos de erro conforme o número de amostras aumenta, dividida em erro de treinamento e erro de validação.")
                st.write(
                    "**Avaliação:** O modelo ideal apresenta uma redução no erro de validação à medida que mais dados são usados para treinamento, com a curva de erro de validação convergindo para a curva de erro de treinamento.")
                fig_learning_curve = self.evaluation_visualization_helper.plot_learning_curve(model, x_train, y_train)
                st.pyplot(fig_learning_curve)

            with col6:
                st.write("### Importância das Features")
                st.write(
                    "**Descrição:** Exibe a contribuição de cada feature no processo de tomada de decisão do modelo.")
                st.write(
                    "**Avaliação:** Features com maior importância têm mais impacto nas previsões do modelo. Avalie se as features mais importantes fazem sentido para o problema em questão.")
                fig_feature_importance = \
                    self.evaluation_visualization_helper.plot_feature_importance(model, x_test, y_test)
                st.pyplot(fig_feature_importance)

            col7, col8 = st.columns(2)

            with col7:
                st.write("### Histograma de Probabilidades")
                st.write(
                    "**Descrição:** Mostra a distribuição das probabilidades previstas para a classe positiva (1).")
                st.write(
                    "**Avaliação:** Um modelo bem calibrado deve ter uma distribuição clara, com alta probabilidade para a classe positiva e baixa para a negativa.")
                fig_probability_histogram = self.evaluation_visualization_helper.plot_probability_histogram(y_proba)
                st.pyplot(fig_probability_histogram)

            with col8:
                st.write("### Estatística de Kappa")
                st.write(
                    "**Descrição:** Mede o grau de concordância entre as previsões do modelo e os valores reais, ajustando para concordâncias ao acaso.")
                st.write(
                    "**Avaliação:** O valor de Kappa varia de -1 a 1, onde 1 indica perfeita concordância, 0 indica concordância ao acaso, e valores negativos indicam desempenho abaixo do esperado.")
                fig_kappa_statistic = self.evaluation_visualization_helper.plot_kappa_statistic(y_test, y_pred)
                st.pyplot(fig_kappa_statistic)
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Matriz de Confusão")
                st.write(
                    "**Descrição:** Mostra a relação entre as previsões do modelo e os valores reais, organizando os dados em Verdadeiros Positivos (TP), Verdadeiros Negativos (TN), Falsos Positivos (FP) e Falsos Negativos (FN).")
                st.write(
                    "**Avaliação:** Idealmente, TP e TN devem ser altos e FP e FN baixos, indicando que o modelo acerta a maioria das previsões.")
                fig_confusion_matrix = self.evaluation_visualization_helper.plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig_confusion_matrix)

            with col2:
                st.write("### Curva de Aprendizado")
                st.write(
                    "**Descrição:** Mostra a performance do modelo em termos de erro conforme o número de amostras aumenta, dividida em erro de treinamento e erro de validação.")
                st.write(
                    "**Avaliação:** O modelo ideal apresenta uma redução no erro de validação à medida que mais dados são usados para treinamento, com a curva de erro de validação convergindo para a curva de erro de treinamento.")
                fig_learning_curve = self.evaluation_visualization_helper.plot_learning_curve(model, x_train, y_train)
                st.pyplot(fig_learning_curve)

            col3, col4 = st.columns(2)

            with col3:
                st.write("### Importância das Features")
                st.write(
                    "**Descrição:** Exibe a contribuição de cada feature no processo de tomada de decisão do modelo.")
                st.write(
                    "**Avaliação:** Features com maior importância têm mais impacto nas previsões do modelo. Avalie se as features mais importantes fazem sentido para o problema em questão.")
                fig_feature_importance = \
                    self.evaluation_visualization_helper.plot_feature_importance(model, x_test, y_test)
                st.pyplot(fig_feature_importance)

            with col4:
                st.write("### Estatística de Kappa")
                st.write(
                    "**Descrição:** Mede o grau de concordância entre as previsões do modelo e os valores reais, ajustando para concordâncias ao acaso.")
                st.write(
                    "**Avaliação:** O valor de Kappa varia de -1 a 1, onde 1 indica perfeita concordância, 0 indica concordância ao acaso, e valores negativos indicam desempenho abaixo do esperado.")
                fig_kappa_statistic = self.evaluation_visualization_helper.plot_kappa_statistic(y_test, y_pred)
                st.pyplot(fig_kappa_statistic)
