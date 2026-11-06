from datasets import load_dataset
import re


def extraer_preguntas(texto, min_palabras=4):
    """
    Busca dentro de un texto (puede ser un párrafo con varias oraciones, como
    un artículo de Wikipedia, o una sola línea, como un subtítulo) fragmentos
    que contengan una pregunta completa '¿...?' de al menos min_palabras palabras.

    Separamos de forma simple por puntos o saltos de línea para aislar oraciones;
    si el texto ya es una sola línea (sin puntos), el split no hace nada y se
    evalúa el texto entero. Devuelve la lista de fragmentos encontrados.
    """
    encontradas = []
    for fragmento in re.split(r'[.\n]', texto):
        fragmento_limpio = fragmento.strip()
        if '¿' in fragmento_limpio and '?' in fragmento_limpio and len(fragmento_limpio.split()) >= min_palabras:
            encontradas.append(fragmento_limpio)
    return encontradas


def _acumular_estandar_y_preguntas(items, extraer_texto, n_estandar, n_preguntas,min_len=0, min_palabras_pregunta=4, incluir_origen=False, prefijo_origen="fuente"):
    """
    Arma dos listas de texto a partir de un dataset en streaming, sin saber
    nada de su esquema (Wikipedia, OpenSubtitles, o lo que venga después):
      1. textos_estandar: los primeros n_estandar textos de más de min_len
         caracteres, tal cual.
      2. textos_preguntas: fragmentos con una pregunta completa '¿...?',
         minados de los textos que no se usaron como estándar (un texto
         corto, aunque no entre al bucket estándar, igual puede aportar
         preguntas), hasta completar n_preguntas.
    Corta en cuanto las dos cuotas están completas.

    items: iterable de filas crudas, tal cual las entrega
        datasets.load_dataset(..., streaming=True) al iterarlo (p. ej. un
        dict por fila). Cada fuente tiene su propia forma de fila (Wikipedia
        trae 'text'; OpenSubtitles trae 'translation' con un dict por idioma),
        así que esta función no intenta leerla directamente.
    extraer_texto: función que recibe una fila de 'items' y devuelve el
        string de texto ya extraído y sin espacios de sobra (p. ej.
        lambda item: item['text'].strip()). Es lo único que cada llamador
        tiene que adaptar a la forma de su propio dataset.
    """
    textos_estandar = []
    textos_preguntas = []

    

    for indice_fila, item in enumerate(items):
        texto = extraer_texto(item)

        origen_id = f"{prefijo_origen}:{indice_fila}"

        def empaquetar(fragmento):
            if incluir_origen:
                return {
                    "texto": fragmento,
                    "origen_id": origen_id
                }
            return fragmento

        if len(textos_estandar) < n_estandar and len(texto) > min_len:
            textos_estandar.append(empaquetar(texto))

        elif len(textos_preguntas) < n_preguntas:
            for pregunta in extraer_preguntas(
                texto, min_palabras=min_palabras_pregunta
            ):
                textos_preguntas.append(empaquetar(pregunta))

                if len(textos_preguntas) >= n_preguntas:
                    break

        # Detenerse cuando ya tenemos suficientes ejemplos
        if (
            len(textos_estandar) >= n_estandar
            and len(textos_preguntas) >= n_preguntas
        ):
            break

    return textos_estandar, textos_preguntas



def cargar_textos_wikipedia(n_estandar=100, n_preguntas=300, min_len=100, aleatorio=False, seed=101, version="20231101.es", incluir_origen = False):
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

    textos_estandar, textos_preguntas = _acumular_estandar_y_preguntas(
        dataset, lambda item: item['text'].strip(), n_estandar, n_preguntas, min_len=min_len, incluir_origen=incluir_origen, prefijo_origen="wikipedia")
    
    print(f"✅ Descarga finalizada: {len(textos_estandar)} textos estándar y {len(textos_preguntas)} oraciones de preguntas obtenidas.")
    return textos_estandar, textos_preguntas


def cargar_opensubtitles(n_estandar=0, n_preguntas=300, min_len=20, min_palabras_pregunta=4, seed=101, idioma="es", incluir_origen =False):
    """
    Descarga líneas de diálogo en español del corpus de subtítulos
    homersimpson/opensubtitles_es (alineado es-ca, en Parquet; no ejecuta
    ningún script, a diferencia de open_subtitles/Helsinki-NLP/open_subtitles,
    que en versiones nuevas de `datasets` ya no se puede cargar).

    A diferencia de Wikipedia, cada fila ya es una línea corta de diálogo
    (no un artículo con varias oraciones), así que no hace falta partir
    nada para armar los textos estándar: se usan tal cual.

    Devuelve (textos_estandar, textos_preguntas), con la misma forma que
    cargar_textos_wikipedia, para poder mezclar ambas fuentes directamente.
    """
    print("Iniciando descarga en streaming de OpenSubtitles en español...")
    dataset = load_dataset("homersimpson/opensubtitles_es", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    textos_estandar, textos_preguntas = _acumular_estandar_y_preguntas(
        dataset, lambda item: item['translation'][idioma].strip(), n_estandar, n_preguntas,
        min_len=min_len, min_palabras_pregunta=min_palabras_pregunta, incluir_origen=incluir_origen,
    prefijo_origen="opensubtitles"
    )
    print(f"✅ Descarga finalizada: {len(textos_estandar)} líneas estándar y {len(textos_preguntas)} líneas con pregunta obtenidas.")
    return textos_estandar, textos_preguntas
