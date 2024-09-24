import streamlit as st

from src.app_components.controllers import ModelController
from src.visualization import EdaVisualizationHelper


class EdaPage:
    def __init__(self, model_controller: ModelController, eda_visualization_helper: EdaVisualizationHelper):
        self.df = model_controller.df
        self.eda_visualization_helper = eda_visualization_helper

    def display(self):
        # Criar três colunas e usar a coluna do meio para centralizar a imagem
        col1, col2, col3 = st.columns([1, 2, 1])

        # Exibir a imagem na coluna do meio
        with col1:
            st.text("")
            st.text("")
            st.image('image/asteroides.png', width=150)

        st.title("Análise Exploratória de Asteroides")
        st.text("")
        st.text("")
        st.text("")
        st.text("")

        col9, col10 = st.columns(2)

        with col9:
            st.write("### Distribuição de Asteroides Potencialmente Perigosos")
            st.write(
                "**Descrição:** Este gráfico mostra a proporção de asteroides que são considerados potencialmente perigosos versus os que não são.")
            st.write(
                "**Avaliação:** A análise ajuda a entender o equilíbrio das classes, o que é importante para avaliar se o modelo está lidando com classes desbalanceadas.")
            fig_target_dist = self.eda_visualization_helper.plot_target_distribution(self.df)
            st.pyplot(fig_target_dist)

        with col10:
            st.write("### Matriz de Correlação")
            st.write(
                "**Descrição:** A matriz de correlação mostra a relação entre as variáveis numéricas do dataset. Cada célula contém o coeficiente de correlação entre dois atributos.")
            st.write(
                "**Avaliação:** Correlações altas (positivas ou negativas) podem indicar redundância de features, o que pode ser problemático para alguns modelos.")
            fig_corr_matrix = self.eda_visualization_helper.plot_correlation_matrix(self.df)
            st.pyplot(fig_corr_matrix)

        col11, col12 = st.columns(2)

        with col11:
            st.write("### Distribuição dos Diâmetros por Periculosidade")
            st.write(
                "**Descrição:** Este gráfico exibe a distribuição do diâmetro dos asteroides em relação à sua periculosidade.")
            st.write(
                "**Avaliação:** A visualização ajuda a verificar se asteroides maiores têm maior probabilidade de serem considerados perigosos.")
            fig_diameter_density = self.eda_visualization_helper.plot_diameter_density(self.df)
            st.pyplot(fig_diameter_density)

        with col12:
            st.write("### Importância das Características")
            st.write("**Descrição:** Exibe a importância relativa das features para o modelo.")
            st.write(
                "**Avaliação:** Identifica quais características mais influenciam a classificação dos asteroides como potencialmente perigosos ou não.")
            fig_feature_importance = self.eda_visualization_helper.plot_feature_importance_eda(self.df)
            st.pyplot(fig_feature_importance)

        col13, col14 = st.columns(2)

        with col13:
            st.write("### Dispersão entre Distância de Aproximação e Velocidade Relativa")
            st.write(
                "**Descrição:** Gráfico de dispersão entre a distância de aproximação mínima e a velocidade relativa dos asteroides.")
            st.write(
                "**Avaliação:** Permite verificar se há uma relação entre a distância de aproximação e a velocidade, ajudando a identificar padrões de comportamento entre asteroides mais próximos.")
            fig_scatter = self.eda_visualization_helper.plot_scatter_miss_distance_velocity(self.df)
            st.pyplot(fig_scatter)

        with col14:
            st.write("### Distribuição Temporal das Aproximações")
            st.write("**Descrição:** Mostra a distribuição das datas em que os asteroides fazem aproximações à Terra.")
            st.write(
                "**Avaliação:** Ajuda a entender a periodicidade ou se há tendências temporais no comportamento dos asteroides.")
            fig_temporal_dist = self.eda_visualization_helper.plot_temporal_distribution(self.df)
            st.pyplot(fig_temporal_dist)

        col15, col16 = st.columns(2)

        with col15:
            st.write("### Distribuição das Variáveis Numéricas")
            st.write("**Descrição:** Exibe a distribuição das principais variáveis numéricas no dataset.")
            st.write(
                "**Avaliação:** Permite observar a distribuição dos dados e identificar outliers, assim como assimetrias.")
            fig_num_distributions = self.eda_visualization_helper.plot_numerical_distributions(self.df)
            st.pyplot(fig_num_distributions)

        with col16:
            st.write("### Verificação de Multicolinearidade (VIF)")
            st.write("**Descrição:** O Variance Inflation Factor (VIF) mede a multicolinearidade entre as features.")
            st.write(
                "**Avaliação:** Um VIF elevado (acima de 10) pode indicar problemas de multicolinearidade, sugerindo que algumas variáveis podem ser redundantes e devem ser removidas.")
            vif_data = self.eda_visualization_helper.calculate_vif(self.df)
            st.dataframe(vif_data)
