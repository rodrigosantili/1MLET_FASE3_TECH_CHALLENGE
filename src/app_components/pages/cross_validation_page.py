import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

from src.app_components.controllers import ModelController


class CrossValidationPage:
    def __init__(self, model_controller: ModelController):
        self.model_controller = model_controller

    def display(self):
        # Criar três colunas e usar a coluna do meio para centralizar a imagem
        col1, col2, col3 = st.columns([1, 2, 1])

        # Exibir a imagem na coluna do meio
        with col1:
            st.text("")
            st.text("")
            st.image('image/asteroides.png', width=150)

        selected_model_name = st.session_state.selected_model_name
        model = self.model_controller.load_selected_model(selected_model_name)
        _, _, x_train, x_test, y_train, y_test = self.model_controller.load_data()

        st.title(f"Validação Cruzada (K-Fold)")

        st.write(f"Realizando validação cruzada para o modelo: {selected_model_name}")

        # Realiza a validação cruzada com base no modelo selecionado
        f1_score = self.model_controller.get_f1_score(selected_model_name, x_train, y_train)

        if not f1_score:
            st.error("Modelo selecionado não encontrado.")
            return

        # Exibe o resultado da validação cruzada
        st.write(f"Modelo {selected_model_name} - F1 Score (ponderado) da Validação Cruzada: {f1_score:.4f}")

        st.title(f"Resultados de Previsão com Dados de Teste")

        # Fazer previsões com os dados de teste
        y_pred_rf = model.predict(x_test)

        # Calcular a acurácia e o classification report
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf)

        # Exibir uma mensagem com o modelo escolhido
        st.write(f"{selected_model_name} - Accuracy: {accuracy_rf}")

        # Exibir uma mensagem com o modelo escolhido
        st.text(f"{selected_model_name} - Classification Report:\n{report_rf}")
