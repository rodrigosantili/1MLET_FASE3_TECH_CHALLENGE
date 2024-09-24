import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app_components.controllers import ModelController
from src.app_components.pages import PredictionPage, ModelEvaluationPage, EdaPage, CrossValidationPage, PcaAnalysisPage
from src.visualization import EvaluationVisualizationHelper, EdaVisualizationHelper


def main():
    st.set_page_config(page_title="Previsão de Asteroides", layout="wide")

    # Initialize components
    model_controller = ModelController()
    eda_visualization_helper = EdaVisualizationHelper()
    evaluation_visualization_helper = EvaluationVisualizationHelper()

    # Load model options
    model_options = model_controller.model_options

    # Model selection
    selected_model_name = st.selectbox("Selecione o modelo para fazer a previsão:", options=model_options.keys())
    st.session_state.selected_model_name = selected_model_name

    # Display selected model message
    st.write(f"Modelo {selected_model_name} carregado com sucesso.")
    st.markdown("---")

    # Display pages
    prediction_page = PredictionPage(model_controller)
    model_evaluation_page = ModelEvaluationPage(model_controller, evaluation_visualization_helper)
    eda_page = EdaPage(model_controller, eda_visualization_helper)
    cross_validation_page = CrossValidationPage(model_controller)
    pca_analysis_page = PcaAnalysisPage(model_controller)

    # Navigation
    st.sidebar.title("Menu de Navegação")
    page = st.sidebar.selectbox("Escolha a página",
                                ["Previsão", "Avaliação do Modelo", "Análise Exploratória EDA", "PCA",
                                 "Validação Cruzada"])

    if page == "Previsão":
        prediction_page.display()
    elif page == "Avaliação do Modelo":
        model_evaluation_page.display()
    elif page == "Análise Exploratória EDA":
        eda_page.display()
    elif page == "Validação Cruzada":
        cross_validation_page.display()
    elif page == "PCA":
        pca_analysis_page.display()


if __name__ == '__main__':
    main()
