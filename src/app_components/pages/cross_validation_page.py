import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

from .page import Page
from src.app_components.controllers import ModelController
from src.utils.string_utils import strings


class CrossValidationPage(Page):
    def __init__(self, model_controller: ModelController):
        self.model_controller = model_controller

    def display(self):
        super().display()

        selected_model_name = st.session_state.selected_model_name
        model = self.model_controller.load_selected_model(selected_model_name)
        _, _, x_train, x_test, y_train, y_test = self.model_controller.load_data()

        st.title(strings["cross_validation_page_title"])

        st.write(strings["cross_validation_model"].format(selected_model_name))

        # Realiza a validação cruzada com base no modelo selecionado
        f1_score = self.model_controller.get_f1_score(selected_model_name, x_train, y_train)

        if not f1_score:
            st.error(strings["cross_validation_model_not_found"])
            return

        # Exibe o resultado da validação cruzada
        st.write(strings["cross_validation_f1_score"].format(selected_model_name, f1_score))

        st.title(strings["cross_validation_prediction_results_title"])

        # Fazer previsões com os dados de teste
        y_pred_rf = model.predict(x_test)

        # Calcular a acurácia e o classification report
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf)

        # Exibir uma mensagem com o modelo escolhido
        st.write(strings["cross_validation_accuracy"].format(selected_model_name, accuracy_rf))

        # Exibir uma mensagem com o modelo escolhido
        st.text(strings["cross_validation_classification_report"].format(selected_model_name, report_rf))
