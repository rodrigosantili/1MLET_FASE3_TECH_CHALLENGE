import streamlit as st

from src.data import SpotifyRecommendationSystem


class StreamlitInterface:
    def __init__(self, recommendation_system: SpotifyRecommendationSystem):
        self.recommendation_system = recommendation_system

    def display_recommendations(self) -> None:
        st.title("Music Recommendation System with Spotify Data")

        search_option = st.radio("Choose the search type:", ("By Song", "By Band/Artist"))

        if search_option == "By Song":
            self._search_by_song()
        elif search_option == "By Band/Artist":
            self._search_by_artist()

    def _search_by_song(self) -> None:
        song_name = st.text_input("Enter the song name:")
        if not song_name:
            return

        st.write(f"Searching for recommendations for the song: {song_name}")
        results = self.recommendation_system.recommend_songs([{'name': song_name}])
        self._display_results(results, song_name)

    def _search_by_artist(self) -> None:
        artist_name = st.text_input("Enter the band or artist name:")
        if not artist_name:
            return

        st.write(f"Searching for recommendations for the band/artist: {artist_name}")
        results = self.recommendation_system.recommend_songs([{'artist': artist_name}])
        self._display_results(results, artist_name)

    def _display_results(self, results, search_term) -> None:
        if results is None or results.empty:
            st.write(f"No recommendations found for '{search_term}'.")
        else:
            st.write(results)
