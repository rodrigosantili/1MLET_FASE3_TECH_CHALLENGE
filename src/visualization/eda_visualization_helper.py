import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


class EdaVisualizationHelper:
    @staticmethod
    def plot_target_distribution(df) -> plt:
        """
        Plot the distribution of the target variable
        :param df: The DataFrame containing the data
        :return: The target distribution plot
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x='is_potentially_hazardous_asteroid', data=df, hue='is_potentially_hazardous_asteroid', legend=False)
        plt.title('Distribution of Potentially Hazardous Asteroids')
        plt.xlabel('Potentially Hazardous Asteroid')
        plt.ylabel('Count')
        return plt

    @staticmethod
    def plot_numerical_distributions(df) -> plt:
        """
        Plot histograms of numerical variables
        :param df: The DataFrame containing the data
        :return: The numerical distributions plot
        """
        df.hist(bins=20, figsize=(15, 10))
        plt.tight_layout()
        return plt

    @staticmethod
    def plot_correlation_matrix(df) -> plt:
        """
        Generate the correlation matrix
        :param df: The DataFrame containing the data
        :return: The correlation matrix plot
        """
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_df = df[numerical_columns]
        plt.figure(figsize=(10, 6))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        return plt

    @staticmethod
    def plot_diameter_density(df) -> plt:
        """
        Plot the density of diameters by hazard level
        :param df: The DataFrame containing the data
        :return: The diameter density plot
        """
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df[df['is_potentially_hazardous_asteroid']]['estimated_diameter_min_km'],
                    label='Hazardous - Min', fill=True, color='red')
        sns.kdeplot(df[df['is_potentially_hazardous_asteroid']]['estimated_diameter_max_km'],
                    label='Hazardous - Max', fill=True, color='blue')
        plt.title('Density Distribution of Minimum and Maximum Diameters by Hazard Level')
        plt.xlabel('Diameter in Kilometers')
        plt.ylabel('Density')
        plt.legend()
        return plt

    @staticmethod
    def calculate_vif(df) -> pd.DataFrame:
        """
        Calculate and display the Variance Inflation Factor (VIF)
        :param df: The DataFrame containing the data
        :return: The VIF DataFrame
        """
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_df = df[numerical_columns].copy()
        numerical_df = sm.add_constant(numerical_df)
        vif_data = pd.DataFrame()
        vif_data['feature'] = numerical_df.columns
        vif_data['VIF'] = [variance_inflation_factor(numerical_df.values, i) for i in range(numerical_df.shape[1])]
        return vif_data

    @staticmethod
    def plot_feature_importance_eda(df) -> plt:
        """
        Plot the feature importance using RandomForest (EDA)
        :param df: The DataFrame containing the data
        :return: The feature importance plot
        """
        X = df.drop('is_potentially_hazardous_asteroid', axis=1)
        y = df['is_potentially_hazardous_asteroid']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='dodgerblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance (EDA)')
        plt.gca().invert_yaxis()
        return plt

    @staticmethod
    def plot_scatter_miss_distance_velocity(df) -> plt:
        """
        Plot the scatter plot between miss_distance_km and relative_velocity_kms
        :param df: The DataFrame containing the data
        :return: The scatter plot
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='miss_distance_km', y='relative_velocity_kms', hue='is_potentially_hazardous_asteroid', data=df)
        plt.title('Scatter Plot between Miss Distance and Relative Velocity')
        return plt

    @staticmethod
    def plot_temporal_distribution(df) -> plt:
        """
        Plot the temporal distribution of close approaches
        :param df: The DataFrame containing the data
        :return: The temporal distribution plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df['epoch_date_close_approach'], bins=100)
        plt.title('Temporal Distribution of Asteroid Approaches')
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        return plt
