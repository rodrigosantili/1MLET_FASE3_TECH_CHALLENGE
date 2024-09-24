# src/app_components/pages/page.py
import os.path
from abc import ABC, abstractmethod
import streamlit as st

from src.config import IMAGES_DIR


class Page(ABC):
    @abstractmethod
    def display(self):
        # Criar três colunas e usar a coluna do meio para centralizar a imagem
        col1, col2, col3 = st.columns([1, 2, 1])

        # Exibir a imagem na coluna do meio
        with col1:
            st.text("")
            st.text("")
            st.image(os.path.abspath(os.path.join(IMAGES_DIR, 'asteroides.png')), width=150)
