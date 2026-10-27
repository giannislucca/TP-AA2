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


def procesar_parrafo_a_chunks(texto_crudo, tokenizer, chunk_size=32, id_inicio=1):
    """
    Genera etiquetas y divide el párrafo en chunks de hasta chunk_size
    tokens, sin separar los sub-tokens de una misma palabra.

    Una palabra de más de chunk_size sub-tokens queda sola en un chunk más largo.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size debe ser un entero positivo")

    # El orden importa: normalizar, separar signos pegados y recién
    # después dividir por espacios. Los signos se guardan como etiquetas.
    texto = separar_puntuacion_pegada(normalizar_signos(texto_crudo))
    palabras = texto.split()

    # Cada elemento contiene TODOS los sub-tokens de una palabra.
    # Así el empaquetado en chunks respeta sus límites.
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
                grupo.append({
                    'token_id': st_id,
                    'token': stok,
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

    # No descartamos chunks cortos: pueden contener puntuaciones importantes.
    if chunk_actual:
        guardar_chunk(chunk_actual, instancia_id)
        instancia_id += 1

    columnas = ['instancia_id', 'token_id', 'token',
                'punt_inicial', 'punt_final', 'capitalización']
    return pd.DataFrame(filas, columns=columnas), instancia_id


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
