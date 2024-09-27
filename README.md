# NASA Data Analysis with API Integration

This project integrates with the NASA API to collect and analyze space-related data. The objective of this notebook is to explore and visualize data obtained from NASA's API, applying various machine learning algorithms to uncover insights. 

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dependencies](#dependencies)
5. [Contributing](#contributing)

## Project Overview

This project fetches data from NASA's API using an API key and processes it for analysis. The data is then pre-processed and fed into various machine learning models such as Random Forest, SVM, and Neural Networks. The models are evaluated for their performance, and visualizations are generated using `matplotlib` and `seaborn`.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone git@github.com:rodrigosantili/1MLET_FASE3_TECH_CHALLENGE.git
   cd 1MLET_FASE3_TECH_CHALLENGE
   ```

2. **Install the dependencies**:
   You can install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your NASA API key**:
   Create a **.env** file with the **API_KEY_NASA** key followed by the password for your NASA API key.
   If you do not have an account to retrieve this data, you can create one here **https://api.nasa.gov/**.   

## Usage

1. Run the file ```python .\src\main.py``` to generate the models


2. Now run the command below in the terminal to launch the application
```
streamlit run .\src\app.py
```
3. Open the local streamlit application in the browser
```
http://localhost:8501
``` 

## Dependencies

This project uses the following Python libraries:

- `requests==2.32.3`
- `pandas==2.2.3`
- `scikit-learn==1.5.2`
- `matplotlib==3.9.2`
- `seaborn==0.13.2`
- `statsmodels==0.14.3`
- `xgboost==2.1.1`
- `streamlit==1.38.0`
- `python-dotenv==1.0.1`
- `joblib~=1.4.2`
- `numpy~=2.1.1`

You can install all dependencies by running the command:
```bash
pip install -r requirements.txt
```

## Contributors (Group 43)

- Bruno Machado Corte Real
- Pedro Henrique Romaoli Garcia
- Rodrigo Santili Sgarioni