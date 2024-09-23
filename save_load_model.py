import joblib
import os


# Função para salvar o modelo com joblib
def save_model_joblib(model, filename='trained_model.joblib'):
    # Define o caminho do diretório onde o modelo será salvo
    models_dir = 'models'
    # Cria o diretório se ele não existir
    os.makedirs(models_dir, exist_ok=True)
    # Monta o caminho completo do arquivo
    model_path = os.path.join(models_dir, filename)
    # Salva o modelo
    joblib.dump(model, model_path)
    print(f"Modelo salvo como {model_path}")


# Função para carregar o modelo salvo com joblib
def load_model_joblib(filename='trained_model.joblib'):
    # Define o caminho do diretório onde o modelo está salvo
    models_dir = 'models'
    # Monta o caminho completo do arquivo
    model_path = os.path.join(models_dir, filename)
    # Carrega o modelo
    model = joblib.load(model_path)
    print(f"Modelo carregado de {model_path}")
    return model
