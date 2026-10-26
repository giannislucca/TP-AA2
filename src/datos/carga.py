from datasets import load_dataset


def cargar_wikipedia(n_textos=100, min_len=100, aleatorio=False, seed=42, version="20231101.es"):
    """
    Descarga en streaming artículos de Wikipedia en español (streaming=True permite bajar filas
    sin descargar gigabytes enteros) y devuelve los primeros n_textos con más de min_len caracteres.

    Si aleatorio=True, mezcla el orden de los artículos en vez de tomar los primeros seguidos
    """
    dataset = load_dataset("wikimedia/wikipedia", version, split="train", streaming=True)

    if aleatorio:
        # mezcla el orden de los shards y además toma al azar de un buffer de 10000 artículos
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    textos = []
    for item in dataset:
        # Nos quedamos con textos no vacíos y de longitud adecuada
        texto = item['text'].strip()
        if len(texto) > min_len:
            textos.append(texto)
        if len(textos) >= n_textos:
            break

    return textos
