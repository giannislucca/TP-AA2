import re
import numpy as np
import torch
import pandas as pd
from tqdm.notebook import tqdm

# Signos que no son etiquetas: los convertimos en una etiqueta o los eliminamos


#Lo sig es descicion de diseño!!!!! Hoy por hoy se reemplazan signos que no usamos por lo de arriba,
#Pero se podria directamente descartar las oraciones que tienen estos signos.
#(VER)

#Estos reemplazos son discutibles, pero seguro que hay que sacar cosas como (;) del texto de wikipedia
REEMPLAZOS = [
    ("…", "."), ("...", "."),
    (";", ","), (":", ","),
    ("!", "."), ("¡", ""),
]
def normalizar_signos(texto):
    """
    Normaliza los signos de puntuación en el texto cambiandolos por otros más simples o eliminándolos.
    """
    for viejo, nuevo in REEMPLAZOS:
        texto = texto.replace(viejo, nuevo)
    return texto
#(VER ARRIBA)




def seccionar_palabra(palabra):
    """
    Separa una palabra cruda en (p_init, partes, p_fin).
    'partes' son las sub-palabras limpias (sin ningún signo salvo el apóstrofo interno).
    """
    # prefijo/sufijo = todo lo que no es letra o dígito en los bordes (comillas, paréntesis, signos)
    i = 0
    while i < len(palabra) and not palabra[i].isalnum():
        i += 1
    j = len(palabra)
    while j > i and not palabra[j - 1].isalnum():
        j -= 1
    prefijo, nucleo, sufijo = palabra[:i], palabra[i:j], palabra[j:]

    # Es lo unico que tenemos considerado como prefijo en terminos de puntuacion
    p_init = "¿" if "¿" in prefijo else ""

    if "?" in sufijo:            # '?' tiene prioridad: ej. '¿qué?».'
        p_fin = "?"
    else:
        #Recorro sufijo de derecha a izquierda y me lo guardo si es coma o punto, si hay varios me quedo con el último, DESICION DE DISEÑO!!!!(VER)
        posibles_sufijos = [c for c in sufijo if c in ",."]
        p_fin = posibles_sufijos[-1] if posibles_sufijos else "" #agarro el utlimo y si no hay nada no agarro nada je

    # guiones y barras internos separan palabras (Castilla-La -> Castilla La),
    # el resto de símbolos internos se descarta (EE.UU -> EEUU, 1.000 -> 1000)
    partes = []

    for palabra in re.split(r"[-–—/]", nucleo): #esto es regex para separar guiones (esos 3 guiones son diferentes- – —) o barra
        limpia = "".join(c for c in palabra if c.isalnum() or c == "'") #dejo el simbolo solo si es un apostrofe, si no lo saco con el join ""
        # basicaemtne hace -> EE.UU -> E E . U U -> descarto . en las itreaciones -> uno E E U U con el join -> EEUU
        if limpia: #si tengo un string no vacio, lo agrego
            partes.append(limpia)

    return p_init, partes, p_fin


def determinar_capitalizacion(palabra_original):
    """
    Determina la clase de capitalización (0, 1, 2, 3) según la consigna.
    """
    letras = [c for c in palabra_original if c.isalpha()] #solo nos interesan las letras porque estamos viendo mayusculas.
    if not letras:
        return 0

    cadena = "".join(letras)
    if cadena.isupper():
        return 1 if len(cadena) == 1 else 3  # 1 si es letra sola  3 si es sigla ESTA DECISION ES A DEBATIR <-------------------------
    if cadena[0].isupper():
        return 2 if any(c.isupper() for c in cadena[1:]) else 1 # 2 si es mixta (ej: 'McDonald'), 1 si es estándar
    if any(c.isupper() for c in cadena):
        return 2 # 2 si es mixta (ej: 'iPhone')
    return 0


def separar_puntuacion_pegada(texto):
    """
    Inserta espacios si un signo de puntuación separa palabras pegadas.
    No borramos puntuacion: lo usamos despues.

    Conserva abreviaturas formadas por grupos cortos (EE.UU, U.S.A)
    y números decimales (1.000) como unidades, tal como hacía el código.
    """
    # Protegemos los puntos internos de abreviaturas: "EE.UU" no es
    # equivalente a "EE. UU". El punto final de "EE.UU." NO se protege.
    abreviaturas = re.finditer(r"(?<![\w.])(?:[A-ZÁÉÍÓÚÜÑ]{1,3}\.)+[A-ZÁÉÍÓÚÜÑ]{1,3}(?![a-záéíóúüñ])", texto)
    puntos_protegidos = {
        i for abrev in abreviaturas
        for i in range(abrev.start(), abrev.end())
        if texto[i] == "."
    }

    # Los puntos entre letras pueden marcar el inicio de una oración,
    # incluso sin mayúscula: "hola.mundo" -> "hola. mundo".
    def separar_punto(match):
        return "." if match.start() in puntos_protegidos else ". "

    texto = re.sub(r"(?<=[^\W\d_])\.(?=[^\W\d_])", separar_punto, texto)
    # "Hola.¿Cómo" -> "Hola. ¿Cómo".
    texto = re.sub(r"(?<=[.,?])(?=¿)", " ", texto)
    # "estás?Bien" -> "estás? Bien".
    texto = re.sub(r"(?<=\?)(?=[^\W\d_])", " ", texto)
    # "Hola,Franco" -> "Hola, Franco" (pero no divide "1,000").
    texto = re.sub(r"(?<=,)(?=[^\W\d_])", " ", texto)
    # "Hola¿cómo" -> "Hola ¿cómo".
    texto = re.sub(r"(?<=[^\W\d_])(?=¿)", " ", texto)
    return texto


def pegar_puntuacion_suelta(texto):
    """
    Pega los signos que quedaron sueltos entre espacios a la palabra que
    acompañan (ej. '¿ Vienes ?' -> '¿Vienes?'), algo raro en Wikipedia pero
    común en subtítulos. Sin esto, seccionar_palabra recibe un '¿' o un '?'
    como palabra propia, sin ninguna letra que le sirva de ancla, y no tiene
    a qué pegarle la etiqueta: la pierde en silencio.
    """
    texto = re.sub(r'¿\s+', '¿', texto)          # '¿ Vienes' -> '¿Vienes'
    texto = re.sub(r'\s+([?,.])', r'\1', texto)  # 'Vienes ?' -> 'Vienes?'
    return texto


def procesar_parrafo_a_chunks(texto_crudo, tokenizer, embedding_fn, chunk_size=32, id_inicio=1):
    """
    Genera etiquetas, calcula embeddings por sub-token y divide el párrafo en chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size debe ser un entero positivo")

    texto = separar_puntuacion_pegada(pegar_puntuacion_suelta(normalizar_signos(texto_crudo)))
    palabras = texto.split()

    grupos_palabras = []

    for palabra in palabras:
        p_init, partes, p_fin = seccionar_palabra(palabra)

        for k, parte in enumerate(partes):
            p_init_parte = p_init if k == 0 else ""
            p_fin_parte = p_fin if k == len(partes) - 1 else ""

            cap_val = determinar_capitalizacion(parte)
            sub_tokens = tokenizer.tokenize(parte.lower())
            if not sub_tokens:
                continue

            sub_token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
            grupo = []
            for i, (stok, st_id) in enumerate(zip(sub_tokens, sub_token_ids)):
                
                # ----------------------------------------------------------------------------
                # Agrego en esta función el cálculo de embeddings, así no lo repetimos después
                # ----------------------------------------------------------------------------
                vec = embedding_fn(stok)
                if vec is not None and isinstance(vec, torch.Tensor):
                    vec = vec.detach().cpu().numpy()
                elif vec is None:
                    # En caso de token fuera de vocabulario, padding con ceros de dim 768
                    vec = np.zeros(768)

                grupo.append({
                    'token_id': st_id,
                    'token': stok,
                    'embedding': vec,  # <-- Columna de embedding por token
                    'punt_inicial': p_init_parte if i == 0 else "",
                    'punt_final': p_fin_parte if i == len(sub_tokens) - 1 else "",
                    'capitalización': cap_val,
                })
            grupos_palabras.append(grupo)

    filas = []
    instancia_id = id_inicio
    chunk_actual = []

    def guardar_chunk(chunk, identificador):
        for token in chunk:
            filas.append({'instancia_id': identificador, **token})

    for grupo in grupos_palabras:
        if chunk_actual and len(chunk_actual) + len(grupo) > chunk_size:
            guardar_chunk(chunk_actual, instancia_id)
            instancia_id += 1
            chunk_actual = []

        chunk_actual.extend(grupo)

    if chunk_actual:
        guardar_chunk(chunk_actual, instancia_id)
        instancia_id += 1

    columnas = ['instancia_id', 'token_id', 'token', 'embedding',
                'punt_inicial', 'punt_final', 'capitalización']
    return pd.DataFrame(filas, columns=columnas), instancia_id


def construir_dataset(textos, tokenizer, embedding_fn, chunk_size=32,
                      exigir_origen=False):

    dataframes_chunks = []
    current_id = 1

    with torch.no_grad():
        for indice_texto, elemento in enumerate(
            tqdm(textos, desc="Construyendo dataset y embeddings por token")
        ):

            # Obtener el texto y su origen
            if isinstance(elemento, dict):
                texto = elemento["texto"]
                origen_id = elemento["origen_id"]
            else:
                if exigir_origen:
                    raise ValueError("Falta origen_id en los textos")
                texto = elemento
                origen_id = f"sin_origen:{indice_texto}"

            # Preprocesamiento original
            df_chunk, current_id = procesar_parrafo_a_chunks(
                texto,
                tokenizer,
                embedding_fn=embedding_fn,
                chunk_size=chunk_size,
                id_inicio=current_id
            )

            if not df_chunk.empty:
                df_chunk["origen_id"] = origen_id
                df_chunk["texto_id"] = indice_texto
                dataframes_chunks.append(df_chunk)

    return pd.concat(dataframes_chunks, ignore_index=True)


def construir_embeddings_por_palabra(df):
    """
    Toma el DataFrame que ya contiene la columna 'embedding' en cada token,
    agrupa los sub-tokens por palabra y calcula la media de sus vectores.
    """
    df_temp = df.copy()
    df_temp['es_inicio_palabra'] = ~df_temp['token'].str.startswith('##')
    df_temp['palabra_id'] = df_temp.groupby('instancia_id')['es_inicio_palabra'].cumsum()
    
    registros_palabras = []
    grupos = df_temp.groupby(['instancia_id', 'palabra_id'])

    for (inst_id, p_id), grupo in tqdm(grupos, desc="Promediando embeddings por palabra"):
        sub_tokens = grupo['token'].tolist()
        vectores_subtokens = grupo['embedding'].tolist()  # Toma los vectores existentes
        
        if not vectores_subtokens:
            continue
            
        embedding_promedio = np.mean(vectores_subtokens, axis=0)
        
        registros_palabras.append({
            'origen_id': grupo.iloc[0]['origen_id'],
            'texto_id': grupo.iloc[0]['texto_id'],
            'instancia_id': inst_id,
            'palabra': "".join([t.replace('##', '') for t in sub_tokens]),
            'embedding': embedding_promedio,
            'punt_inicial': grupo.iloc[0]['punt_inicial'],
            'punt_final': grupo.iloc[-1]['punt_final'],
            'capitalización': grupo.iloc[0]['capitalización']
        })

    return pd.DataFrame(registros_palabras)

###Constructores de features a mano (QUEDA MEDIO FEO ACA, REPENSAR ESTRUCTURA)###
VOCALES = set("aeiouáéíóúü")
VOCALES_CON_TILDE = set("áéíóú")

# Pronombres/adverbios interrogativos: fuerte señal de punt_inicial='¿' en español
INTERROGATIVOS = {
    "qué", "quién", "quiénes", "cómo", "cuándo", "dónde",
    "cuál", "cuáles", "cuánto", "cuánta", "cuántos", "cuántas",
}
# Nexos típicos antes de coma o punto
NEXOS = {"y", "pero", "porque", "que", "entonces", "luego", "aunque", "además"}
# Artículos y preposiciones más frecuentes: suelen ir sin puntuación
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "en", "a", "con", "por", "para", "sin", "sobre",
}


def _calcular_features_palabra(palabra):
    """
    Atributos "a mano" de una palabra (ya en minúsculas, sin puntuación),
    sin usar embeddings.
    """
    letras = [c for c in palabra if c.isalpha()]
    n_letras = len(letras)
    n_vocales = sum(1 for c in letras if c in VOCALES)
    n_tildes = sum(1 for c in letras if c in VOCALES_CON_TILDE)

    return np.array([
        len(palabra),
        n_letras,
        n_vocales,
        n_letras - n_vocales,          # consonantes
        n_tildes,
        float(palabra in INTERROGATIVOS),
        float(palabra in NEXOS),
        float(palabra in STOPWORDS),
        float(palabra.isdigit()),
    ])


def construir_features_manuales_por_palabra(df):
    """
    Análogo a construir_embeddings_por_palabra para el sub-ítem a.ii (sin
    embeddings): agrupa sub-tokens en palabras igual que esa función, pero
    en vez de promediar embeddings de BERT calcula _calcular_features_palabra.
    """
    df_temp = df.copy()
    df_temp['es_inicio_palabra'] = ~df_temp['token'].str.startswith('##')
    df_temp['palabra_id'] = df_temp.groupby('instancia_id')['es_inicio_palabra'].cumsum()

    registros_palabras = []
    grupos = df_temp.groupby(['instancia_id', 'palabra_id'])

    for (inst_id, p_id), grupo in tqdm(grupos, desc="Procesando features manuales de palabras"):#tqdm es para mostrar barra de progreso visual
        sub_tokens = grupo['token'].tolist()
        palabra = "".join(t.replace('##', '') for t in sub_tokens)#rearmo la palabra original

        registros_palabras.append({
            'origen_id': grupo.iloc[0]['origen_id'],
            'texto_id': grupo.iloc[0]['texto_id'],
            'instancia_id': inst_id,
            'palabra': palabra,
            'features': _calcular_features_palabra(palabra),
            'punt_inicial': grupo.iloc[0]['punt_inicial'],
            'punt_final': grupo.iloc[-1]['punt_final'],
            'capitalización': grupo.iloc[0]['capitalización']
        })

    return pd.DataFrame(registros_palabras)


def generar_matriz_features_rf(df_palabras, window_size=2, columna_features='embedding'):
    """
    Genera la matriz X e y para entrenar el RF concatenando la ventana de contexto
    (palabras a la izquierda y derecha) y los vectores y para cada target.

    columna_features: nombre de la columna de df_palabras con el vector por
    palabra (embeddings o atributos a mano)
    """
    X, y_init, y_fin, y_cap = [], [], [], []
    dim_emb = len(df_palabras.iloc[0][columna_features])#Aca se cambio a columna de features generica
    padding_vec = np.zeros(dim_emb)  # Vector de ceros para los bordes del chunk

    # Iterar por cada chunk (instancia_id) por separado para no cruzar palabras de distintas oraciones
    for _, chunk in df_palabras.groupby('instancia_id'):
        embeddings_chunk = chunk[columna_features].tolist() #Aca se cambio a columna de features generica
        p_init_chunk = chunk['punt_inicial'].tolist()
        p_fin_chunk = chunk['punt_final'].tolist()
        cap_chunk = chunk['capitalización'].tolist()

        n_palabras = len(embeddings_chunk)

        for i in range(n_palabras):
            vector_ventana = []
            
            # Palabras a la izquierda (vecinos anteriores)
            for j in range(i - window_size, i):
                if j >= 0:
                    vector_ventana.append(embeddings_chunk[j])
                else:
                    vector_ventana.append(padding_vec)
                    
            # Palabra actual
            vector_ventana.append(embeddings_chunk[i])
            
            # Palabras a la derecha (vecinos posteriores)
            for j in range(i + 1, i + window_size + 1):
                if j < n_palabras:
                    vector_ventana.append(embeddings_chunk[j])
                else:
                    vector_ventana.append(padding_vec)
                    
            # Concatenamos todo en un único vector continuo para esta palabra
            X.append(np.concatenate(vector_ventana))
            y_init.append(p_init_chunk[i])
            y_fin.append(p_fin_chunk[i])
            y_cap.append(cap_chunk[i])

    return np.array(X), np.array(y_init), np.array(y_fin), np.array(y_cap)


def preparar_secuencias_rnn(df):
    """
    Organiza el dataset etiquetado por token para una RNN many-to-many.

    Una fila del DataFrame representa un subtoken. Agrupa por instancia_id,
    conserva el orden original de los subtokens y agrega padding al final.

    Devuelve un diccionario con:
      X            (B, T, D) float32: embeddings estáticos de BERT
      y_init       (B, T) long: puntuación inicial (""=0, "¿"=1)
      y_fin        (B, T) long: puntuación final (""=0, ","=1, "."=2, "?"=3)
      y_cap        (B, T) long: capitalización (0, 1, 2, 3)
      mascara      (B, T) bool: True en tokens reales, False en padding
      longitudes   (B,) long: cantidad de tokens reales por secuencia
      instancia_ids: lista de IDs en el mismo orden que las secuencias

    Las etiquetas de padding se marcan con -100, compatible con
    torch.nn.CrossEntropyLoss(ignore_index=-100).
    """
    columnas = ["instancia_id", "embedding", "punt_inicial",
                "punt_final", "capitalización"]

    # Convierte las etiquetas a números
    mapa_init = {"": 0, "¿": 1}
    mapa_fin = {"": 0, ",": 1, ".": 2, "?": 3}

    # Agrupa los tokens en secuencias
    grupos = list(df.groupby("instancia_id", sort=False))
    B = len(grupos)
    T = max(len(grupo) for _, grupo in grupos)
    D = len(np.asarray(df.iloc[0]["embedding"]))

    # Crea los tensores vacíos
    X = np.zeros((B, T, D), dtype=np.float32)
    y_init = np.full((B, T), -100, dtype=np.int64)
    y_fin = np.full((B, T), -100, dtype=np.int64)
    y_cap = np.full((B, T), -100, dtype=np.int64)
    mascara = np.zeros((B, T), dtype=bool)
    longitudes = np.zeros(B, dtype=np.int64)
    instancia_ids = []

    # Recorre cada secuencia y recupera sus embeddings
    for i, (inst_id, grupo) in enumerate(grupos):
        longitud = len(grupo)
        vectores = np.stack(grupo["embedding"].to_list()).astype(np.float32)
        if vectores.shape != (longitud, D) or not np.isfinite(vectores).all():
            raise ValueError(f"Embeddings inválidos en instancia_id={inst_id}")

        # Valida y transforma las etiquetas
        etiquetas_init = grupo["punt_inicial"].map(mapa_init)
        etiquetas_fin = grupo["punt_final"].map(mapa_fin)
        etiquetas_cap = pd.to_numeric(grupo["capitalización"], errors="coerce")
        if (etiquetas_init.isna().any() or etiquetas_fin.isna().any()
                or not etiquetas_cap.isin([0, 1, 2, 3]).all()):
            raise ValueError(f"Etiquetas inválidas en instancia_id={inst_id}")

        # Llena los tensores
        X[i, :longitud] = vectores
        y_init[i, :longitud] = etiquetas_init.to_numpy(dtype=np.int64)
        y_fin[i, :longitud] = etiquetas_fin.to_numpy(dtype=np.int64)
        y_cap[i, :longitud] = etiquetas_cap.to_numpy(dtype=np.int64)
        mascara[i, :longitud] = True
        longitudes[i] = longitud
        instancia_ids.append(inst_id)

    # Convierte a tensores de PyTorch
    return {
        "X": torch.from_numpy(X),
        "y_init": torch.from_numpy(y_init),
        "y_fin": torch.from_numpy(y_fin),
        "y_cap": torch.from_numpy(y_cap),
        "mascara": torch.from_numpy(mascara),
        "longitudes": torch.from_numpy(longitudes),
        "instancia_ids": instancia_ids,
    }
