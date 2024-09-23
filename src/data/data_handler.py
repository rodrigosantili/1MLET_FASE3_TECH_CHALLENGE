import pandas as pd


class DataHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
        self.data['year'] = self.data['year'].fillna(self.data['year'].mean())
