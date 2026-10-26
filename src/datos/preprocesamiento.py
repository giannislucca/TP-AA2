import re

import pandas as pd

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


def procesar_parrafo_a_chunks(texto_crudo, tokenizer, chunk_size=32, id_inicio=1):
    """
    Procesa un párrafo crudo, extrae las etiquetas reales a nivel token y
    lo divide en chunks de tamaño fijo (chunk_size).
    """
    palabras = normalizar_signos(texto_crudo).split()  # CAMBIO: antes texto_crudo.split()

    tokens_totales = []
    token_ids_totales = []
    p_init_totales = []
    p_fin_totales = []
    cap_totales = []

    for palabra in palabras:
        # CAMBIO: antes se miraba solo si empezaba con ¿ o terminaba con , . ?
        p_init, partes, p_fin = seccionar_palabra(palabra)

        # cada parte (ej. 'Castilla' y 'La' de 'Castilla-La') se trata como una palabra:
        # el '¿' va a la primera parte y la puntuación final a la última
        for k, parte in enumerate(partes):
            #Asigno las puntuaicones solo si son la primer o ultima parte (esto capaz cambiar)
            p_init_parte = p_init if k == 0 else ""
            p_fin_parte = p_fin if k == len(partes) - 1 else ""

            cap_val = determinar_capitalizacion(parte)  #usamos la función de anter para determinar la capitalización de la palabra
            palabra_norm = parte.lower()

            # Tokenizar versión normalizada con BERT
            sub_tokens = tokenizer.tokenize(palabra_norm)
            sub_token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)

            if not sub_tokens:
                continue

            # Asignar etiquetas a sub-tokens según reglas del TP (esto lo saque del ejemplo de lo que tiene que devolver el modelo que está en el enunciado)
            for i, (stok, st_id) in enumerate(zip(sub_tokens, sub_token_ids)):
                #primero guardamos tokens y ids en nuestras listas que definimos arriba
                tokens_totales.append(stok)
                token_ids_totales.append(st_id)

                # Puntuación inicial al primer sub-token, final al último
                p_init_totales.append(p_init_parte if i == 0 else "")
                p_fin_totales.append(p_fin_parte if i == len(sub_tokens) - 1 else "")
                cap_totales.append(cap_val) #capaz suena raro pero la capitalización hay que ponersela a cada subtoken

    # Partir en bloques de tamaño 'chunk_size' (ej. 32 tokens)
    filas = []
    instancia_id = id_inicio

    for i in range(0, len(tokens_totales), chunk_size):
        chunk_toks = tokens_totales[i : i + chunk_size]

        if len(chunk_toks) < 5:  # Descartar remanentes insignificantes
            continue

        chunk_ids = token_ids_totales[i : i + chunk_size]
        chunk_p_init = p_init_totales[i : i + chunk_size]
        chunk_p_fin = p_fin_totales[i : i + chunk_size]
        chunk_cap = cap_totales[i : i + chunk_size]

        #acá definimos para cada sub-token las 3 etiquetas de los 3 problema de clasificación que tenemos (es nada más juntar toda la info que recolectamos antes)
        for t, tid, pi, pf, c in zip(chunk_toks, chunk_ids, chunk_p_init, chunk_p_fin, chunk_cap):
            filas.append({
                'instancia_id': instancia_id,
                'token_id': tid,
                'token': t,
                'punt_inicial': pi,
                'punt_final': pf,
                'capitalización': c
            })
        instancia_id += 1

    return pd.DataFrame(filas), instancia_id


def construir_dataset(textos, tokenizer, chunk_size=32):
    """
    Aplica procesar_parrafo_a_chunks a cada texto y concatena todo en el
    dataset tabular final con la estructura exigida en el TP.
    """
    dataframes_chunks = []
    current_id = 1

    for parrafo in textos:
        df_chunk, current_id = procesar_parrafo_a_chunks(parrafo, tokenizer, chunk_size=chunk_size, id_inicio=current_id)
        if not df_chunk.empty:
            dataframes_chunks.append(df_chunk)

    return pd.concat(dataframes_chunks, ignore_index=True) #concatenamos todos los datasets por parrafo que disponemos.
