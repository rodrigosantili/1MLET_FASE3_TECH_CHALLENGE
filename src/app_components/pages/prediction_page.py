import numpy as np
import streamlit as st

from src.app_components.controllers import ModelController
from src.utils.time_utils import convert_year_to_epoch


class PredictionPage:
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

        st.title("Previsão de Asteroides Potencialmente Perigosos")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        # Criar colunas para os campos de entrada
        col1, col2 = st.columns(2)

        with col1:
            # Magnitude Absoluta H
            st.write("**Magnitude Absoluta H**:")
            st.dialog("Refere-se ao brilho intrínseco do asteroide.")
            absolute_magnitude_h = st.number_input('Magnitude Absoluta H (min 9.25, max 30.1)', value=16.57)

            # Diâmetro Estimado Máximo
            st.write("**Diâmetro Estimado Máximo (km)**: ")
            st.dialog("Estimativa do maior diâmetro do asteroide.")
            estimated_diameter_max_km = st.number_input('Diâmetro Estimado Máximo (km) (min 0.0056, max 83.95)',
                                                        value=2.884297)

            # Velocidade Relativa
            st.write("**Velocidade Relativa (km/s)**:")
            st.dialog("Velocidade do asteroide em relação à Terra.")
            relative_velocity_kms = st.number_input('Velocidade Relativa (km/s) (min 0.075, max 94.57)',
                                                    value=27.008071)

            # Distância de Passagem
            st.write("**Distância de Passagem (km)**:")
            st.dialog("Distância mínima da Terra durante a aproximação do asteroide.")
            miss_distance_km = st.number_input('Distância de Passagem (km) (min 12913.14, max 299134700)',
                                               value=12638584.486877)

            # Ano de Aproximação
            st.write("**Ano de Aproximação**:")
            st.dialog("Ano da aproximação mais próxima em formato epoch.")
            year = st.number_input('Ano de Aproximação (min 1900, max 2100)', value=2024, min_value=1900,
                                   max_value=2100)
            epoch_date_close_approach = convert_year_to_epoch(year)

            # Tipo de Classe de Órbita
            st.write("**Tipo de Classe de Órbita**:")
            st.dialog("Define a classe de órbita do asteroide. Valores: (0=Não classificado, 1=Amor, 2=Apollo, 3=Aten)")
            orbit_class_type = st.number_input('Tipo de Classe de Órbita (0, 1, 2, 3)', value=1.0)

        with col2:
            # Excentricidade
            st.write("**Excentricidade**:")
            st.dialog("Mede a forma da órbita (0 = circular, 1 = extremamente alongada).")
            eccentricity = st.number_input('Excentricidade (min 0.013, max 0.968)', value=0.826952)

            # Distância Periélio
            st.write("**Distância Periélio (AU)**:")
            st.dialog("Distância mínima do asteroide ao Sol em unidades astronômicas.")
            perihelion_distance = st.number_input('Distância Periélio (AU) (min 0.0704, max 1.299)', value=0.186553)

            # Inclinação
            st.write("**Inclinação (graus)**:")
            st.dialog("Inclinação da órbita do asteroide em relação ao plano da eclíptica.")
            inclination = st.number_input('Inclinação (graus) (min 0.054, max 154.35)', value=22.80351)

            # Mínima Interseção de Órbita
            st.write("**Mínima Interseção de Órbita (AU)**:")
            st.dialog("Distância mínima entre a órbita do asteroide e a da Terra.")
            minimum_orbit_intersection = st.number_input('Mínima Interseção de Órbita (min 0.000025, max 0.708)',
                                                         value=0.034215)

            # Período Orbital
            st.write("**Período Orbital (dias)**:")
            st.dialog("Tempo que o asteroide leva para completar uma órbita ao redor do Sol.")
            orbital_period = st.number_input('Período Orbital (dias) (min 176.57, max 27441.14)', value=408.837519)

            # Invariante de Tisserand de Júpiter
            st.write("**Invariante de Tisserand de Júpiter**:")
            st.dialog("Parâmetro dinâmico que mede a influência gravitacional de Júpiter no asteroide.")
            jupiter_tisserand_invariant = st.number_input('Invariante de Tisserand de Júpiter (min 1.316, max 9.039)',
                                                          value=5.299)

        # Prever com base nas entradas
        if st.button('Prever'):
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
                st.title(f"Previsão: {'Asteroide Perigoso' if prediction == 1 else 'Asteroide Não Perigoso'}")
            except Exception as e:
                st.error(f"Erro ao processar a previsão: {e}")
