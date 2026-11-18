"""
Inferencia: predicciones por token para un CSV en el formato del enunciado.

Entrada: CSV con columnas instancia_id, token_id, token (las columnas a predecir
pueden venir vacías o no estar). Salida: el mismo CSV, con las mismas filas y en
el mismo orden, con punt_inicial, punt_final y capitalización completas.

Reglas de formato (enunciado):
  - Una palabra empieza en cada token que no comienza con "##".
  - punt_inicial se asigna solo al primer token de la palabra.
  - punt_final se asigna solo al último token de la palabra.
  - capitalización es la misma para todos los tokens de la palabra.
Para cada palabra usamos la predicción del modelo en el primer token (inicial y
capitalización) y en el último token (final), igual que en la evaluación por palabra.
"""
import numpy as np
import pandas as pd
import torch

MAPA_INICIAL = np.array(["", "¿"], dtype=object)
MAPA_FINAL = np.array(["", ",", ".", "?"], dtype=object)
DIM_EMBEDDING = 768


def leer_csv_tokens(ruta):
    """
    Lee el CSV sin convertir nada a NaN: con la lectura por defecto, pandas
    transformaría tokens como "nan" o "null" en valores faltantes.
    """
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    faltan = {"instancia_id", "token_id", "token"} - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {faltan}")
    return df


def tabla_embeddings(tokens, embedding_fn):
    """
    Embedding estático de cada token distinto, con la misma función que usamos
    para construir el dataset de entrenamiento. Los tokens fuera de vocabulario
    reciben un vector de ceros, como en el entrenamiento.
    """
    tabla = {}
    with torch.no_grad():
        for tok in pd.unique(tokens):
            vec = embedding_fn(tok)
            if vec is None:
                tabla[tok] = np.zeros(DIM_EMBEDDING, dtype=np.float32)
            else:
                tabla[tok] = vec.detach().cpu().numpy().astype(np.float32)
    return tabla


def predecir_por_token(modelo, df, embedding_fn, batch_size=64, device=None):
    """
    Devuelve tres arrays (una predicción por fila de df) con las clases
    numéricas de punt_inicial, punt_final y capitalización.

    Cada instancia se procesa completa, sin partirla en chunks: la red recurrente
    acepta secuencias de cualquier largo. El padding se arma por batch, al largo
    de la instancia más larga del batch, y el modelo lo ignora mediante las longitudes.
    """
    if device is None:
        device = next(modelo.parameters()).device
    modelo.eval()

    tabla = tabla_embeddings(df["token"], embedding_fn)

    # Posiciones (filas de df) de cada instancia, en el orden en que aparecen
    grupos = [idx.to_numpy() for _, idx in
              df.groupby("instancia_id", sort=False).groups.items()]
    grupos = [np.sort(g) for g in grupos]

    pred_ini = np.zeros(len(df), dtype=np.int64)
    pred_fin = np.zeros(len(df), dtype=np.int64)
    pred_cap = np.zeros(len(df), dtype=np.int64)
    tokens = df["token"].to_numpy()

    with torch.no_grad():
        for inicio in range(0, len(grupos), batch_size):
            lote = grupos[inicio:inicio + batch_size]
            longitudes = torch.tensor([len(g) for g in lote], dtype=torch.long)
            X = np.zeros((len(lote), int(longitudes.max()), DIM_EMBEDDING), dtype=np.float32)
            for i, filas in enumerate(lote):
                X[i, :len(filas)] = np.stack([tabla[t] for t in tokens[filas]])

            logits_i, logits_f, logits_c = modelo(torch.from_numpy(X).to(device), longitudes)
            p_i = logits_i.argmax(dim=-1).cpu().numpy()
            p_f = logits_f.argmax(dim=-1).cpu().numpy()
            p_c = logits_c.argmax(dim=-1).cpu().numpy()

            for i, filas in enumerate(lote):
                n = len(filas)
                pred_ini[filas] = p_i[i, :n]
                pred_fin[filas] = p_f[i, :n]
                pred_cap[filas] = p_c[i, :n]

    return pred_ini, pred_fin, pred_cap


def aplicar_formato_por_palabra(df, pred_ini, pred_fin, pred_cap):
    """
    Pasa de predicciones por token a etiquetas con el formato del enunciado
    (ver reglas al principio del archivo). Devuelve una copia de df con las
    tres columnas completas.
    """
    salida = df.copy()
    es_inicio = ~salida["token"].str.startswith("##").to_numpy()
    instancia = salida["instancia_id"].to_numpy()
    # Una palabra también termina donde cambia la instancia
    nueva_instancia = np.r_[True, instancia[1:] != instancia[:-1]]
    es_inicio = es_inicio | nueva_instancia
    es_final = np.r_[es_inicio[1:], True]

    # Índice de la palabra a la que pertenece cada token, y su primer y último token
    palabra = np.cumsum(es_inicio) - 1
    primer_token = np.flatnonzero(es_inicio)
    ultimo_token = np.flatnonzero(es_final)

    punt_inicial = np.full(len(salida), "", dtype=object)
    punt_final = np.full(len(salida), "", dtype=object)
    punt_inicial[primer_token] = MAPA_INICIAL[pred_ini[primer_token]]
    punt_final[ultimo_token] = MAPA_FINAL[pred_fin[ultimo_token]]
    capitalizacion = pred_cap[primer_token][palabra]

    salida["punt_inicial"] = punt_inicial
    salida["punt_final"] = punt_final
    salida["capitalización"] = capitalizacion.astype(int)
    return salida[["instancia_id", "token_id", "token",
                   "punt_inicial", "punt_final", "capitalización"]]


def verificar_formato(entrada, salida):
    """Chequeos del archivo de salida antes de entregarlo. Lanza AssertionError si algo falla."""
    assert len(salida) == len(entrada), "Cambió la cantidad de filas"
    for col in ["instancia_id", "token_id", "token"]:
        assert (salida[col].astype(str).to_numpy() == entrada[col].astype(str).to_numpy()).all(), \
            f"Cambió el orden o el contenido de la columna {col}"
    assert set(salida["punt_inicial"]) <= {"", "¿"}
    assert set(salida["punt_final"]) <= {"", ",", ".", "?"}
    assert set(salida["capitalización"].astype(int)) <= {0, 1, 2, 3}

    sub = salida["token"].str.startswith("##").to_numpy()
    assert (salida["punt_inicial"].to_numpy()[sub] == "").all(), "Hay ¿ en un token que no inicia palabra"
    sigue_sub = np.r_[sub[1:], False] & (salida["instancia_id"].to_numpy()
                                         == np.r_[salida["instancia_id"].to_numpy()[1:], None])
    assert (salida["punt_final"].to_numpy()[sigue_sub] == "").all(), "Hay puntuación final en medio de una palabra"
    cap = salida["capitalización"].astype(int).to_numpy()
    assert (cap[1:][sub[1:]] == cap[:-1][sub[1:]]).all(), "La capitalización cambia dentro de una palabra"
    return True


def generar_predicciones_csv(modelo, ruta_entrada, ruta_salida, embedding_fn, batch_size=64):
    """Lee el CSV de entrada, predice, aplica el formato, verifica y escribe el CSV de salida."""
    entrada = leer_csv_tokens(ruta_entrada)
    pred_ini, pred_fin, pred_cap = predecir_por_token(modelo, entrada, embedding_fn, batch_size)
    salida = aplicar_formato_por_palabra(entrada, pred_ini, pred_fin, pred_cap)
    verificar_formato(entrada, salida)
    # QUOTE_MINIMAL (el valor por defecto) encierra entre comillas los campos con ",", como pide el enunciado
    salida.to_csv(ruta_salida, index=False, encoding="utf-8")
    return salida
