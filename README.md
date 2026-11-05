# TP 2 — Aprendizaje Automático II

Predicción de puntuación y capitalización en texto normalizado. El enunciado está en `TP-2-2025-2C-vNov6.pdf`.

## Estructura

```
TP-AA2/
├── codigoTP.ipynb            # notebook principal: solo llama a las funciones de src/ y muestra resultados
└── src/
    ├── __init__.py
    └── datos/
        ├── __init__.py
        ├── carga.py              # cargar_wikipedia(...)
        ├── preprocesamiento.py   # normalización de signos, etiquetado por token y armado del dataset en chunks
        └── analisis.py           # resumen_etiquetas(df): distribución de etiquetas y chequeo de clases
```

La lógica va en `src/`; la notebook importa y ejecuta. Si se agrega una función nueva, va en el archivo de `src/` que corresponda (o en una carpeta nueva dentro de `src/` si es otra etapa, por ejemplo `src/modelos/`).

## Cómo correrlo en Google Colab

La notebook hace `from src.datos... import ...`, así que **la carpeta `src/` tiene que estar en Colab antes de correr la celda de imports**. Colab trabaja en `/content`, por lo que tiene que quedar `/content/src/...`.

### Opción A: subir `src/` a mano

1. Abrir `codigoTP.ipynb` en Colab.
2. Abrir el panel de archivos (ícono de carpeta en la barra izquierda).
3. Sobre `content`: clic derecho → **Nueva carpeta** → `src`. Dentro de `src`, crear otra carpeta `datos`.
4. Clic derecho sobre cada carpeta → **Subir** y subir sus archivos:
   - en `src/`: `__init__.py`
   - en `src/datos/`: `__init__.py`, `carga.py`, `preprocesamiento.py`, `analisis.py`

   Los `__init__.py` están vacíos pero son necesarios para que funcionen los imports.
   La carpeta `__pycache__` no hace falta subirla.
5. Correr la notebook de arriba hacia abajo.

### Opción B: Colab desde VS Code

1. Abrir el repo local en VS Code.
2. Instalar la extensión de Google Colab y conectar el kernel: arriba a la derecha, elegir kernel → Colab → la primera opción.
3. Con el kernel ya conectado, clic derecho sobre la carpeta `src/` en el explorador de archivos → **Upload to Colab**.
4. Correr la notebook de arriba hacia abajo.

### Cosas a tener en cuenta

- **Los archivos de Colab se borran al reiniciar o desconectar el entorno.** Después de eso hay que volver a subir `src/` (Opción A o B).
- Si se modifica un archivo de `src/` (desde el editor de Colab o subiendo una versión nueva), no hace falta reiniciar: la notebook tiene `%autoreload 2`, que recarga los módulos al ejecutar la próxima celda.
- Los cambios hechos en Colab sobre `src/` **no vuelven al repo** solos: hay que descargarlos y commitearlos.
- Dependencias: `transformers` y `pandas` vienen instalados en Colab. Si falla `from datasets import load_dataset`, correr `!pip install -q datasets` y reiniciar el entorno.

