#En este file poner herramientas de analiss: reports, graficos, prints esquematicos, etc



def resumen_etiquetas(df):
    """
    Imprime el tamaño del dataset, la distribución de cada etiqueta y si aparecen todas las clases.
    """
    print(f"Número total de tokens (filas en CSV): {len(df)}")
    print(f"Número total de instancias (chunks): {df['instancia_id'].nunique()}\n")

    palabras = df[~df["token"].str.startswith("##")]

    print("Punt. inicial:\n", df["punt_inicial"].replace("", "(nada)").value_counts(), "\n")
    print("Punt. final:\n",   df["punt_final"].replace("", "(nada)").value_counts(), "\n")
    print("Capitalización (por palabra):\n", palabras["capitalización"].value_counts().sort_index(), "\n")

    # % de instancias que tienen al menos una pregunta
    con_preg = df.groupby("instancia_id")["punt_final"].apply(lambda s: (s == "?").any())
    print(f"Instancias con '?': {con_preg.mean():.2%}")

    # chequeo: ¿aparecen todas las clases?
    esperadas = {"punt_inicial": {"", "¿"}, "punt_final": {"", ",", ".", "?"}, "capitalización": {0, 1, 2, 3}}
    for col, clases in esperadas.items():
        faltan = clases - set(df[col].unique())
        print(col, "OK" if not faltan else f"FALTAN: {faltan}")
