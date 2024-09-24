import streamlit as st
from src.app_components.controllers import ModelController


class PcaAnalysisPage:
    def __init__(self, model_controller: ModelController):
        self.model_controller = model_controller

    def display(self):
        _, pca, _, _, _, _ = self.model_controller.load_data()

        st.write(f"### Variancia PCA")

        # Exibir a variância explicada
        explained_variance = pca.explained_variance_ratio_
        st.write(
            "**Descrição:** O gráfico de variância explicada pelos componentes principais mostra a quantidade de informação capturada por cada componente.")
        st.write(
            "**Avaliação:** A análise de PCA ajuda a identificar quais componentes principais contêm mais variância nos dados, facilitando a redução dimensional e visualização de características importantes no modelo. Os valores de variância explicada por componente variam de 0 a 1")
        st.text(f"Variância explicada por cada componente: {explained_variance}")
        st.text(f"Variância total explicada: {sum(explained_variance)}")
