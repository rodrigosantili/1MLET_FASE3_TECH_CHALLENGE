import requests
import json
import os


def fetch_asteroids(api_key, max_objects=1000, base_url="https://api.nasa.gov/neo/rest/v1/neo/browse"):
    fetched_objects = []
    page = 0

    try:
        while len(fetched_objects) < max_objects:
            # Montar a URL para a requisição com a página atual
            url = f"{base_url}?page={page}&api_key={api_key}"
            # Fazer a requisição para a API
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'near_earth_objects' in data:
                    # Adicionar os objetos até atingir o limite máximo
                    for obj in data['near_earth_objects']:
                        if len(fetched_objects) < max_objects:
                            fetched_objects.append(obj)
                            if len(fetched_objects) % 1000 == 0:
                                print(f"Total de objetos coletados: {len(fetched_objects)}")
                        else:
                            break
                page += 1  # Ir para a próxima página
            else:
                print(f"Erro ao acessar API: {response.status_code}")
                break
    except Exception as e:
        print(f"Erro durante a requisição: {e}")

    return fetched_objects


def save_to_json(data, filename='asteroids_data.json'):
    download_folder = '.'

    file_path = os.path.join(download_folder, filename)

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Arquivo salvo em: {file_path}")
