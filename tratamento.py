import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
#
#
# def process_asteroid_data_from_json(folder_path='.', filename='fetched_asteroids.json'):
#     # Carregar o arquivo JSON
#     json_file_path = os.path.join(folder_path, filename)
#
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as file:
#             asteroids = json.load(file)
#         print(f"Arquivo {filename} carregado com sucesso!")
#     except FileNotFoundError:
#         print(f"Erro: O arquivo {filename} não foi encontrado no diretório {folder_path}.")
#         return None
#     except ValueError as e:
#         print(f"Erro ao carregar o arquivo JSON: {e}")
#         return None
#
#     # Verificar se os dados estão em formato de lista
#     if not isinstance(asteroids, list):
#         print(f"Erro: O conteúdo do arquivo JSON não é uma lista. Verifique a estrutura do arquivo.")
#         return None
#
#     # Inicializando variáveis
#     asteroid_data = []
#     current_year = datetime.now().year
#     current_epoch = datetime.now().timestamp() * 1000  # Converte a data atual para timestamp em milissegundos
#
#     # Processando os dados para extrair as features relevantes
#     for neo in asteroids:
#         # Verificar se cada objeto 'neo' é um dicionário
#         if not isinstance(neo, dict):
#             print(f"Erro: O objeto NEO não é um dicionário. Verifique os dados.")
#             continue
#
#         close_approach_data = neo.get('close_approach_data', [])
#         for approach in close_approach_data:
#             epoch_date_close_approach = approach.get('epoch_date_close_approach')
#             if epoch_date_close_approach and epoch_date_close_approach <= current_epoch:
#                 features = {
#                     'absolute_magnitude_h': neo.get('absolute_magnitude_h'),
#                     'estimated_diameter_min_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get(
#                         'estimated_diameter_min'),
#                     'estimated_diameter_max_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get(
#                         'estimated_diameter_max'),
#                     'relative_velocity_kms': approach.get('relative_velocity', {}).get('kilometers_per_second'),
#                     'miss_distance_km': approach.get('miss_distance', {}).get('kilometers'),
#                     'epoch_date_close_approach': epoch_date_close_approach,
#                     'orbit_class_type': neo.get('orbital_data', {}).get('orbit_class', {}).get('orbit_class_type'),
#                     'eccentricity': neo.get('orbital_data', {}).get('eccentricity'),
#                     'semi_major_axis': neo.get('orbital_data', {}).get('semi_major_axis'),
#                     'perihelion_distance': neo.get('orbital_data', {}).get('perihelion_distance'),
#                     'inclination': neo.get('orbital_data', {}).get('inclination'),
#                     'minimum_orbit_intersection': neo.get('orbital_data', {}).get('minimum_orbit_intersection'),
#                     'orbital_period': neo.get('orbital_data', {}).get('orbital_period'),
#                     'jupiter_tisserand_invariant': neo.get('orbital_data', {}).get('jupiter_tisserand_invariant'),
#                     'is_potentially_hazardous_asteroid': neo.get('is_potentially_hazardous_asteroid')
#                 }
#                 asteroid_data.append(features)
#
#     # Criar DataFrame
#     df = pd.DataFrame(asteroid_data)
#
#     # Seleção de colunas numéricas para conversão
#     numeric_columns = [
#         'relative_velocity_kms',
#         'miss_distance_km',
#         'eccentricity',
#         'semi_major_axis',
#         'perihelion_distance',
#         'inclination',
#         'minimum_orbit_intersection',
#         'orbital_period',
#         'jupiter_tisserand_invariant'
#     ]
#
#     # Conversão de colunas numéricas
#     for column in numeric_columns:
#         if column in df.columns:
#             df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
#         else:
#             print(f"A coluna '{column}' não foi encontrada no DataFrame.")
#
#     # Converter 'orbit_class_type' para categorias numéricas
#     if 'orbit_class_type' in df.columns:
#         label_encoder = LabelEncoder()
#         df['orbit_class_type'] = label_encoder.fit_transform(df['orbit_class_type'])
#     else:
#         print("A coluna 'orbit_class_type' não foi encontrada no DataFrame.")
#
#     print(df.info())
#
#     return df

def process_asteroid_data_from_json(folder_path='.', filename='fetched_asteroids.json'):
    # Carregar o arquivo JSON
    json_file_path = os.path.join(folder_path, filename)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            asteroids = json.load(file)
        print(f"Arquivo {filename} carregado com sucesso!")
    except FileNotFoundError:
        print(f"Erro: O arquivo {filename} não foi encontrado no diretório {folder_path}.")
        return None
    except ValueError as e:
        print(f"Erro ao carregar o arquivo JSON: {e}")
        return None

    # Verificar se os dados estão em formato de lista
    if not isinstance(asteroids, list):
        print(f"Erro: O conteúdo do arquivo JSON não é uma lista. Verifique a estrutura do arquivo.")
        return None

    # Inicializando variáveis
    asteroid_data = []
    current_year = datetime.now().year
    current_epoch = datetime.now().timestamp() * 1000  # Converte a data atual para timestamp em milissegundos

    # Processando os dados para extrair as features relevantes
    for neo in asteroids:
        # Verificar se cada objeto 'neo' é um dicionário
        if not isinstance(neo, dict):
            print(f"Erro: O objeto NEO não é um dicionário. Verifique os dados.")
            continue

        close_approach_data = neo.get('close_approach_data', [])
        for approach in close_approach_data:
            epoch_date_close_approach = approach.get('epoch_date_close_approach')
            if epoch_date_close_approach and epoch_date_close_approach <= current_epoch:
                features = {
                    'absolute_magnitude_h': neo.get('absolute_magnitude_h'),
                    'estimated_diameter_min_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get(
                        'estimated_diameter_min'),
                    'estimated_diameter_max_km': neo.get('estimated_diameter', {}).get('kilometers', {}).get(
                        'estimated_diameter_max'),
                    'relative_velocity_kms': approach.get('relative_velocity', {}).get('kilometers_per_second'),
                    'miss_distance_km': approach.get('miss_distance', {}).get('kilometers'),
                    'epoch_date_close_approach': epoch_date_close_approach,
                    'orbit_class_type': neo.get('orbital_data', {}).get('orbit_class', {}).get('orbit_class_type'),
                    'eccentricity': neo.get('orbital_data', {}).get('eccentricity'),
                    'semi_major_axis': neo.get('orbital_data', {}).get('semi_major_axis'),
                    'perihelion_distance': neo.get('orbital_data', {}).get('perihelion_distance'),
                    'inclination': neo.get('orbital_data', {}).get('inclination'),
                    'minimum_orbit_intersection': neo.get('orbital_data', {}).get('minimum_orbit_intersection'),
                    'orbital_period': neo.get('orbital_data', {}).get('orbital_period'),
                    'jupiter_tisserand_invariant': neo.get('orbital_data', {}).get('jupiter_tisserand_invariant'),
                    'is_potentially_hazardous_asteroid': neo.get('is_potentially_hazardous_asteroid')
                }
                asteroid_data.append(features)

    # Criar DataFrame
    df = pd.DataFrame(asteroid_data)

    # Seleção de colunas numéricas para conversão
    numeric_columns = [
        'relative_velocity_kms',
        'miss_distance_km',
        'eccentricity',
        'semi_major_axis',
        'perihelion_distance',
        'inclination',
        'minimum_orbit_intersection',
        'orbital_period',
        'jupiter_tisserand_invariant'
    ]

    # Conversão de colunas numéricas
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
        else:
            print(f"A coluna '{column}' não foi encontrada no DataFrame.")

    # Converter 'orbit_class_type' para categorias numéricas
    if 'orbit_class_type' in df.columns:
        label_encoder = LabelEncoder()
        df['orbit_class_type'] = label_encoder.fit_transform(df['orbit_class_type'])
    else:
        print("A coluna 'orbit_class_type' não foi encontrada no DataFrame.")

    return df
