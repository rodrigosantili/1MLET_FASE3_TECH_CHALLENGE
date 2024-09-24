import numpy as np
import streamlit as st

from .page import Page
from src.app_components.controllers import ModelController
from src.utils.string_utils import strings
from src.utils.time_utils import convert_year_to_epoch


class PredictionPage(Page):
    def __init__(self, model_controller: ModelController):
        self.model_controller = model_controller

    def display(self):
        super().display()

        st.title(strings["prediction_page_title"])
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        # Criar colunas para os campos de entrada
        col1, col2 = st.columns(2)

        with col1:
            # Magnitude Absoluta H
            st.write(strings["absolute_magnitude_h"])
            st.dialog(strings["absolute_magnitude_h_dialog"])
            absolute_magnitude_h = st.number_input(strings["absolute_magnitude_h_range"], value=16.57)

            # Diâmetro Estimado Máximo
            st.write(strings["estimated_diameter_max_km"])
            st.dialog(strings["estimated_diameter_max_km_dialog"])
            estimated_diameter_max_km = st.number_input(strings["estimated_diameter_max_km_range"], value=2.884297)

            # Velocidade Relativa
            st.write(strings["relative_velocity_kms"])
            st.dialog(strings["relative_velocity_kms_dialog"])
            relative_velocity_kms = st.number_input(strings["relative_velocity_kms_range"], value=27.008071)

            # Distância de Passagem
            st.write(strings["miss_distance_km"])
            st.dialog(strings["miss_distance_km_dialog"])
            miss_distance_km = st.number_input(strings["miss_distance_km_range"], value=12638584.486877)

            # Ano de Aproximação
            st.write(strings["year_of_approach"])
            st.dialog(strings["year_of_approach_dialog"])
            year = st.number_input(strings["year_of_approach_range"], value=2024, min_value=1900, max_value=2100)
            epoch_date_close_approach = convert_year_to_epoch(year)

            # Tipo de Classe de Órbita
            st.write(strings["orbit_class_type"])
            st.dialog(strings["orbit_class_type_dialog"])
            orbit_class_type = st.number_input(strings["orbit_class_type_range"], value=1.0)

        with col2:
            # Excentricidade
            st.write(strings["eccentricity"])
            st.dialog(strings["eccentricity_dialog"])
            eccentricity = st.number_input(strings["eccentricity_range"], value=0.826952)

            # Distância Periélio
            st.write(strings["perihelion_distance"])
            st.dialog(strings["perihelion_distance_dialog"])
            perihelion_distance = st.number_input(strings["perihelion_distance_range"], value=0.186553)

            # Inclinação
            st.write(strings["inclination"])
            st.dialog(strings["inclination_dialog"])
            inclination = st.number_input(strings["inclination_range"], value=22.80351)

            # Mínima Interseção de Órbita
            st.write(strings["minimum_orbit_intersection"])
            st.dialog(strings["minimum_orbit_intersection_dialog"])
            minimum_orbit_intersection = st.number_input(strings["minimum_orbit_intersection_range"], value=0.034215)

            # Período Orbital
            st.write(strings["orbital_period"])
            st.dialog(strings["orbital_period_dialog"])
            orbital_period = st.number_input(strings["orbital_period_range"], value=408.837519)

            # Invariante de Tisserand de Júpiter
            st.write(strings["jupiter_tisserand_invariant"])
            st.dialog(strings["jupiter_tisserand_invariant_dialog"])
            jupiter_tisserand_invariant = st.number_input(strings["jupiter_tisserand_invariant_range"], value=5.299)

        # Prever com base nas entradas
        if st.button(strings["predict_button"]):
            try:
                features = np.array(
                    [[absolute_magnitude_h, estimated_diameter_max_km, relative_velocity_kms, miss_distance_km,
                      epoch_date_close_approach, orbit_class_type, eccentricity, perihelion_distance,
                      inclination, minimum_orbit_intersection, orbital_period, jupiter_tisserand_invariant]])
                scaler, pca, _, _, _, _ = self.model_controller.load_data()
                scaled_features = scaler.transform(features)
                pca_features = pca.transform(scaled_features)
                model = self.model_controller.load_selected_model(st.session_state.selected_model_name)
                prediction = model.predict(pca_features)[0]
                st.title(strings["prediction_result"].format(
                    strings["dangerous_asteroid"] if prediction == 1 else strings["non_dangerous_asteroid"]))
            except Exception as e:
                st.error(strings["prediction_error"].format(e))
