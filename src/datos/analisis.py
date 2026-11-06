#En este file poner herramientas de analiss: reports, graficos, prints esquematicos, etc
import matplotlib.pyplot as plt
import seaborn as sns



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


def graficar_windows(df_resultados, ventanas):
    """
    Toma el data frame de resultados para diferentes RF variando su window size y realiza un gráfico
    """
    
    # Formatear datos para facilitar la graficación con Seaborn
    df_melted = df_resultados.melt(id_vars='window_size', var_name='Tarea', value_name='F1 Macro')

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.lineplot(
        data=df_melted,
        x='window_size',
        y='F1 Macro',
        hue='Tarea',
        marker='o',
        linewidth=2.5,
        markersize=8
    )

    # Detalles estéticos del gráfico
    plt.title('Impacto del Tamaño de Ventana (Contexto) en el F1-Macro', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Tamaño de Ventana (window_size)', fontsize=12)
    plt.ylabel('F1-Macro', fontsize=12)
    plt.xticks(ventanas)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Agregar los valores numéricos sobre cada marcador
    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        for x, y in zip(x_data, y_data):
            plt.text(x, y + 0.02, f"{y:.3f}", ha='center', va='bottom', fontsize=9, fontweight='semibold')

    plt.legend(title='Tareas', title_fontsize='11', loc='lower right')
    plt.tight_layout()
    plt.show()