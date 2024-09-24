import streamlit as st

from .page import Page
from src.app_components.controllers import ModelController
from src.utils.string_utils import strings
from src.visualization import EvaluationVisualizationHelper


class ModelEvaluationPage(Page):
    def __init__(self,
                 model_controller: ModelController,
                 evaluation_visualization_helper: EvaluationVisualizationHelper):
        self.model_controller = model_controller
        self.evaluation_visualization_helper = evaluation_visualization_helper

    def display(self):
        super().display()

        st.title(strings["model_evaluation_page_title"])
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
                st.write(strings["confusion_matrix_title"])
                st.write(strings["confusion_matrix_description"])
                st.write(strings["confusion_matrix_eval"])
                fig_confusion_matrix = self.evaluation_visualization_helper.plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig_confusion_matrix)

            with col2:
                st.write(strings["roc_curve_title"])
                st.write(strings["roc_curve_description"])
                st.write(strings["roc_curve_eval"])
                fig_roc_curve = self.evaluation_visualization_helper.plot_roc_curve(y_test, y_proba)
                st.pyplot(fig_roc_curve)

            col3, col4 = st.columns(2)

            with col3:
                st.write(strings["pr_curve_title"])
                st.write(strings["pr_curve_description"])
                st.write(strings["pr_curve_eval"])
                fig_pr_curve = self.evaluation_visualization_helper.plot_precision_recall_curve(y_test, y_proba)
                st.pyplot(fig_pr_curve)

            with col4:
                st.write(strings["lift_curve_title"])
                st.write(strings["lift_curve_description"])
                st.write(strings["lift_curve_eval"])
                fig_lift_curve = self.evaluation_visualization_helper.plot_lift_curve(y_test, y_proba)
                st.pyplot(fig_lift_curve)

            col5, col6 = st.columns(2)

            with col5:
                st.write(strings["learning_curve_title"])
                st.write(strings["learning_curve_description"])
                st.write(strings["learning_curve_eval"])
                fig_learning_curve = self.evaluation_visualization_helper.plot_learning_curve(model, x_train, y_train)
                st.pyplot(fig_learning_curve)

            with col6:
                st.write(strings["feature_importance_title"])
                st.write(strings["feature_importance_description"])
                st.write(strings["feature_importance_eval"])
                fig_feature_importance = \
                    self.evaluation_visualization_helper.plot_feature_importance(model, x_test, y_test)
                st.pyplot(fig_feature_importance)

            col7, col8 = st.columns(2)

            with col7:
                st.write(strings["probability_histogram_title"])
                st.write(strings["probability_histogram_description"])
                st.write(strings["probability_histogram_eval"])
                fig_probability_histogram = self.evaluation_visualization_helper.plot_probability_histogram(y_proba)
                st.pyplot(fig_probability_histogram)

            with col8:
                st.write(strings["kappa_statistic_title"])
                st.write(strings["kappa_statistic_description"])
                st.write(strings["kappa_statistic_eval"])
                fig_kappa_statistic = self.evaluation_visualization_helper.plot_kappa_statistic(y_test, y_pred)
                st.pyplot(fig_kappa_statistic)
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.write(strings["confusion_matrix_title"])
                st.write(strings["confusion_matrix_description"])
                st.write(strings["confusion_matrix_eval"])
                fig_confusion_matrix = self.evaluation_visualization_helper.plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig_confusion_matrix)

            with col2:
                st.write(strings["learning_curve_title"])
                st.write(strings["learning_curve_description"])
                st.write(strings["learning_curve_eval"])
                fig_learning_curve = self.evaluation_visualization_helper.plot_learning_curve(model, x_train, y_train)
                st.pyplot(fig_learning_curve)

            col3, col4 = st.columns(2)

            with col3:
                st.write(strings["feature_importance_title"])
                st.write(strings["feature_importance_description"])
                st.write(strings["feature_importance_eval"])
                fig_feature_importance = \
                    self.evaluation_visualization_helper.plot_feature_importance(model, x_test, y_test)
                st.pyplot(fig_feature_importance)

            with col4:
                st.write(strings["kappa_statistic_title"])
                st.write(strings["kappa_statistic_description"])
                st.write(strings["kappa_statistic_eval"])
                fig_kappa_statistic = self.evaluation_visualization_helper.plot_kappa_statistic(y_test, y_pred)
                st.pyplot(fig_kappa_statistic)
