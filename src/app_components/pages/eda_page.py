import streamlit as st

from .page import Page
from src.app_components.controllers import ModelController
from src.utils.string_utils import strings
from src.visualization import EdaVisualizationHelper


class EdaPage(Page):
    def __init__(self, model_controller: ModelController, eda_visualization_helper: EdaVisualizationHelper):
        self.df = model_controller.df
        self.eda_visualization_helper = eda_visualization_helper

    def display(self):
        super().display()

        st.title(strings["eda_page_title"])
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        col9, col10 = st.columns(2)

        with col9:
            st.write(strings["eda_dangerous_asteroids_distribution_title"])
            st.write(strings["eda_dangerous_asteroids_distribution_description"])
            st.write(strings["eda_dangerous_asteroids_distribution_eval"])
            fig_target_dist = self.eda_visualization_helper.plot_target_distribution(self.df)
            st.pyplot(fig_target_dist)

        with col10:
            st.write(strings["eda_correlation_matrix_title"])
            st.write(strings["eda_correlation_matrix_description"])
            st.write(strings["eda_correlation_matrix_eval"])
            fig_corr_matrix = self.eda_visualization_helper.plot_correlation_matrix(self.df)
            st.pyplot(fig_corr_matrix)

        col11, col12 = st.columns(2)

        with col11:
            st.write(strings["eda_diameter_density_title"])
            st.write(strings["eda_diameter_density_description"])
            st.write(strings["eda_diameter_density_eval"])
            fig_diameter_density = self.eda_visualization_helper.plot_diameter_density(self.df)
            st.pyplot(fig_diameter_density)

        with col12:
            st.write(strings["eda_feature_importance_title"])
            st.write(strings["eda_feature_importance_description"])
            st.write(strings["eda_feature_importance_eval"])
            fig_feature_importance = self.eda_visualization_helper.plot_feature_importance_eda(self.df)
            st.pyplot(fig_feature_importance)

        col13, col14 = st.columns(2)

        with col13:
            st.write(strings["eda_scatter_miss_distance_velocity_title"])
            st.write(strings["eda_scatter_miss_distance_velocity_description"])
            st.write(strings["eda_scatter_miss_distance_velocity_eval"])
            fig_scatter = self.eda_visualization_helper.plot_scatter_miss_distance_velocity(self.df)
            st.pyplot(fig_scatter)

        with col14:
            st.write(strings["eda_temporal_distribution_title"])
            st.write(strings["eda_temporal_distribution_description"])
            st.write(strings["eda_temporal_distribution_eval"])
            fig_temporal_dist = self.eda_visualization_helper.plot_temporal_distribution(self.df)
            st.pyplot(fig_temporal_dist)

        col15, col16 = st.columns(2)

        with col15:
            st.write(strings["eda_numerical_distributions_title"])
            st.write(strings["eda_numerical_distributions_description"])
            st.write(strings["eda_numerical_distributions_eval"])
            fig_num_distributions = self.eda_visualization_helper.plot_numerical_distributions(self.df)
            st.pyplot(fig_num_distributions)

        with col16:
            st.write(strings["eda_vif_title"])
            st.write(strings["eda_vif_description"])
            st.write(strings["eda_vif_eval"])
            vif_data = self.eda_visualization_helper.calculate_vif(self.df)
            st.dataframe(vif_data)
