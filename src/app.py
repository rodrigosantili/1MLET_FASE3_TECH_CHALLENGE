import os
import sys
import streamlit as st

# Incluindo HTML para o uso de ícones no menu lateral
from streamlit.components.v1 import html

# Adicionando o caminho para os componentes do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app_components.controllers import ModelController
from src.app_components.pages import PredictionPage, ModelEvaluationPage, EdaPage, CrossValidationPage, PcaAnalysisPage
from src.visualization import EvaluationVisualizationHelper, EdaVisualizationHelper
from src.utils.string_utils import strings

# Custom CSS to remove borders and adjust the button layout
def inject_custom_css():
    st.markdown("""
        <style>
        .stButton button {
            background-color: transparent !important;
            color: white !important;
            border: none !important;
            box-shadow: none !important;
            padding: 10px 15px;
            text-align: left !important;
            font-size: 18px !important;
        }
        .stButton button:hover {
            background-color: #3d3d3d !important;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=strings["app_page_title"], layout="wide")

    # Inject the custom CSS.
    inject_custom_css()

    # Initialize components
    model_controller = ModelController()
    eda_visualization_helper = EdaVisualizationHelper()
    evaluation_visualization_helper = EvaluationVisualizationHelper()

    # Load model options
    model_options = model_controller.model_options

    # Model selection
    selected_model_name = st.selectbox(strings["model_selection_prompt"], options=model_options.keys())
    st.session_state.selected_model_name = selected_model_name

    # Display selected model message
    st.write(strings["model_loaded_success"].format(selected_model_name))
    st.markdown("---")

    # Display pages
    prediction_page = PredictionPage(model_controller)
    model_evaluation_page = ModelEvaluationPage(model_controller, evaluation_visualization_helper)
    eda_page = EdaPage(model_controller, eda_visualization_helper)
    cross_validation_page = CrossValidationPage(model_controller)
    pca_analysis_page = PcaAnalysisPage(model_controller)

    # Navigation - Creating the menu with buttons and icons in the sidebar.
    st.sidebar.title("Menu")

    if st.sidebar.button('📊 Análise Exploratória (EDA)'):
        st.session_state.page = "eda"
    if st.sidebar.button('🧩 Análise PCA'):
        st.session_state.page = "pca"
    if st.sidebar.button('🔁 Validação Cruzada'):
        st.session_state.page = "cross_validation"
    if st.sidebar.button('📉 Avaliação do Modelo'):
        st.session_state.page = "evaluation"
    if st.sidebar.button('🎯 Previsão'):
        st.session_state.page = "prediction"



    # Check which page was selected.
    if "page" not in st.session_state:
        st.session_state.page = "eda"

    # Render the selected page.
    if st.session_state.page == "prediction":
        prediction_page.display()
    elif st.session_state.page == "evaluation":
        model_evaluation_page.display()
    elif st.session_state.page == "eda":
        eda_page.display()
    elif st.session_state.page == "pca":
        pca_analysis_page.display()
    elif st.session_state.page == "cross_validation":
        cross_validation_page.display()

if __name__ == "__main__":
    main()
