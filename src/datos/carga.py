from datasets import load_dataset
import re

def cargar_textos_wikipedia(n_estandar=100, n_preguntas=300, min_len=100, aleatorio=False, seed=42, version="20231101.es"):
    """
    Descarga textos de Wikipedia en español en streaming.
    Devuelve:
      1. Una lista de textos estándar (artículos completos).
      2. Una lista de ORACIONES individuales que contienen preguntas (¿ ?), 
         lo cual incrementa drásticamente la proporción de etiquetas '?' en el dataset.
    """
    print(f"Iniciando descarga optimizada en streaming de Wikipedia en español (Versión: {version})...")
    dataset = load_dataset("wikimedia/wikipedia", version, split="train", streaming=True)
    
    if aleatorio or n_preguntas > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    
    textos_estandar = []
    textos_preguntas = []
    
    for item in dataset:
        texto = item['text'].strip()
        
        # 1. Obtener textos estándar completos
        if len(textos_estandar) < n_estandar and len(texto) > min_len:
            textos_estandar.append(texto)
            
        # 2. Si aún necesitamos preguntas, dividimos el artículo en oraciones individuales
        elif len(textos_preguntas) < n_preguntas:
            # Separamos de forma simple por puntos o saltos de línea para aislar oraciones
            oraciones = re.split(r'[.\n]', texto)
            for oracion in oraciones:
                oracion_limpia = oracion.strip()
                if '¿' in oracion_limpia and '?' in oracion_limpia and len(oracion_limpia.split()) >= 4:
                    textos_preguntas.append(oracion_limpia)
                    if len(textos_preguntas) >= n_preguntas:
                        break
                        
        else:
            break
            
    print(f"✅ Descarga finalizada: {len(textos_estandar)} textos estándar y {len(textos_preguntas)} oraciones de preguntas obtenidas.")
    return textos_estandar, textos_preguntas