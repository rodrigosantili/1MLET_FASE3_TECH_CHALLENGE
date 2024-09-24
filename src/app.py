import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app_components.controllers import ModelController
from src.app_components.pages import PredictionPage, ModelEvaluationPage, EdaPage, CrossValidationPage, PcaAnalysisPage
from src.visualization import EvaluationVisualizationHelper, EdaVisualizationHelper
from src.utils.string_utils import strings


def main():
    st.set_page_config(page_title=strings["app_page_title"], layout="wide")

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

    # Navigation
    st.sidebar.title(strings["navigation_menu_title"])
    pages = [strings["page_prediction"], strings["page_model_evaluation"], strings["page_eda"],
             strings["page_pca"], strings["page_cross_validation"]]
    selected_page = st.sidebar.selectbox(strings["navigation_menu_title"], pages)

    if selected_page == strings["page_prediction"]:
        prediction_page.display()
    elif selected_page == strings["page_model_evaluation"]:
        model_evaluation_page.display()
    elif selected_page == strings["page_eda"]:
        eda_page.display()
    elif selected_page == strings["page_pca"]:
        pca_analysis_page.display()
    elif selected_page == strings["page_cross_validation"]:
        cross_validation_page.display()


if __name__ == "__main__":
    main()
