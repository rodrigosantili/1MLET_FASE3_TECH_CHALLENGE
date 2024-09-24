import streamlit as st

from .page import Page
from src.app_components.controllers import ModelController
from src.utils.string_utils import strings


class PcaAnalysisPage(Page):
    def __init__(self, model_controller: ModelController):
        self.model_controller = model_controller

    def display(self):
        super().display()

        _, pca, _, _, _, _ = self.model_controller.load_data()

        st.write(strings["pca_variance_title"])

        # Exibir a variância explicada
        explained_variance = pca.explained_variance_ratio_
        st.write(strings["pca_variance_description"])
        st.write(strings["pca_variance_eval"])
        st.text(strings["pca_variance_explained"].format(explained_variance))
