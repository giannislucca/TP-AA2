
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import funciones para el early stopping
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#comentario para ignronates como yo(gianis)
# "Head": es la V de la clase (o<t> = g2(V h<t>)), aplicada sobre el h<t> compartido
# que sale de la GRU/LSTM. En vez de una sola V que predice las C clases de UNA tarea
# (como en el ejemplo de la slide), acá hay 3 V's distintas -una por tarea-, cada una
# con su propio tamaño de salida, pero las tres leen el mismo h<t>.


# Ingnorar padding para gru y lstm. Función auxiliar para poder ignorar las posiciones de padding en la secuencia, que no aportan información.

def ejecutar_recurrente(capa, x, longitudes):
    """
    Ejecuta una GRU o LSTM ignorando las posiciones de padding.
    Devuelve las salidas con las dimensiones originales (B, T, H).
    """

    # 1. Empaquetamos las secuencias indicando sus longitudes reales.
    # Así la RNN no procesa las posiciones de padding.
    # enforce_sorted=False permite recibir secuencias sin ordenarlas por longitud.
    x_packed = pack_padded_sequence(
        x,
        longitudes.detach().cpu(),
        batch_first=True,
        enforce_sorted=False
    )

    # 2. Ejecutamos la GRU o LSTM sobre las secuencias empaquetadas.
    # Nos quedamos con las salidas de todos los tokens (many-to-many).
    # Descartamos los estados finales, que no necesitamos para esta tarea.
    out_packed, _ = capa(x_packed)

    # 3. Reconstruimos el tensor agregando padding al final.
    # total_length conserva la longitud original T para que los
    # tres heads puedan generar predicciones para cada posición.
    out, _ = pad_packed_sequence(
        out_packed,
        batch_first=True,
        total_length=x.size(1)
    )

    # 4. Devolvemos las representaciones contextuales de cada token.
    # Dimensiones: (B, T, H), donde H se duplica si la RNN es bidireccional.
    return out


# GRU multitarea para predicción de puntuación inicial, puntuación final y capitalización
class GRUMultitarea(nn.Module):
    """
    GRU unidireccional multitarea: a partir de embeddings de BERT por sub-token,
    predice punt_inicial, punt_final y capitalización para cada posición de la secuencia.
    """
    def __init__(self, input_size=768, hidden_size=128, dropout=0.2, bidir = False):
        
        super().__init__()

        # input_size=768: dimensión del embedding de BERT que entra en cada posición de la secuencia.
        # hidden_size=128: tamaño del estado recurrente que la GRU va arrastrando token a token.
        # num_layers=1: una sola capa de GRU (no apilamos varias GRUs una arriba de otra).
        # batch_first=True: los tensores que recibe/devuelve tienen forma (B, T, N) y no (T, B, N).
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidir
        )

        # Dropout sobre la salida de la GRU, antes de los heads, para regularizar.
        self.dropout = nn.Dropout(dropout)

        dim_salida_rnn = hidden_size * (2 if bidir else 1)  # si es bidireccional, el tamaño de salida es el doble del hidden_size

        # Una capa de salida por tarea
        # El número de salidas de cada head es la cantidad de clases de esa tarea:
        # punt_inicial tiene 2 clases, punt_final y capitalización tienen 4 cada una.
        self.head_init = nn.Linear(dim_salida_rnn, 2)
        self.head_fin = nn.Linear(dim_salida_rnn, 4)
        self.head_cap = nn.Linear(dim_salida_rnn, 4)

    def forward(self, x, longitudes):
        
        out = ejecutar_recurrente(self.gru, x, longitudes)


        out = self.dropout(out)# esto es h<t> para todo t

        logits_init = self.head_init(out) # o<t> para la tarea 1, vía su propia V (tamaño 2) (sin softmax aplicado, para las de abajo tmb)
        logits_fin = self.head_fin(out) # o<t> para la tarea 2, vía su propia V (tamaño 4)
        logits_cap = self.head_cap(out) # o<t> para la tarea 3, vía su propia V (tamaño 4)

        return logits_init, logits_fin, logits_cap

# LSTM multitarea para predicción de puntuación inicial, puntuación final y capitalización
class LSTMMultitarea(nn.Module):
    """
    LSTM unidireccional multitarea: a partir de embeddings de BERT por sub-token,
    predice punt_inicial, punt_final y capitalización para cada posición de la secuencia.
    """
    def __init__(self, input_size=768, hidden_size=128, dropout=0.2, bidir = False):

        super().__init__()

        # input_size=768: dimensión del embedding de BERT que entra en cada posición de la secuencia.
        # hidden_size=128: tamaño del estado recurrente que la LSTM va arrastrando token a token.
        # num_layers=1: una sola capa de LSTM (no apilamos varias LSTM una arriba de otra).
        # batch_first=True: los tensores que recibe/devuelve tienen forma (B, T, N) y no (T, B, N).
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidir
        )

        # Dropout sobre la salida de la LSTM, antes de los heads, para regularizar.
        self.dropout = nn.Dropout(dropout)

        dim_salida_rnn = hidden_size * (2 if bidir else 1)  # si es bidireccional, el tamaño de salida es el doble del hidden_size
        # Una capa de salida por tarea
        # El número de salidas de cada head es la cantidad de clases de esa tarea:
        # punt_inicial tiene 2 clases, punt_final y capitalización tienen 4 cada una.
        self.head_init = nn.Linear(dim_salida_rnn, 2)
        self.head_fin = nn.Linear(dim_salida_rnn, 4)
        self.head_cap = nn.Linear(dim_salida_rnn, 4)

    def forward(self, x, longitudes):
        
        out = ejecutar_recurrente(self.lstm, x, longitudes)


        out = self.dropout(out)# esto es h<t> para todo t

        logits_init = self.head_init(out) # o<t> para la tarea 1, vía su propia V (tamaño 2) (sin softmax aplicado, para las de abajo tmb)
        logits_fin = self.head_fin(out) # o<t> para la tarea 2, vía su propia V (tamaño 4)
        logits_cap = self.head_cap(out) # o<t> para la tarea 3, vía su propia V (tamaño 4)

        return logits_init, logits_fin, logits_cap


# AMBAS DE ABAJO FUERON RETOCADAS/CREADAS PARA EL PONDERADO
# F de perdida


def calcular_perdida_multitarea(logits_init, logits_fin, logits_cap,
                               y_init, y_fin, y_cap, pesos_clases=None):#pesos_clases=None por defecto 
                                                                        #así  podemos seguir utilizando la función sin ponderaciones, para testeo más que nada.
    """
    Calcula la pérdida conjunta de las tres tareas.

    Si pesos_clases=None, utiliza Cross-Entropy estándar.
    Si recibe pesos, utiliza Weighted Cross-Entropy.
    Ignora siempre las posiciones de padding (-100).
    """

    # 1. Recuperamos los pesos de cada tarea, si fueron proporcionados
    if pesos_clases is not None:
        w_init = pesos_clases["punt_inicial"].to(
            device=logits_init.device, dtype=logits_init.dtype
        )
        w_fin = pesos_clases["punt_final"].to(
            device=logits_fin.device, dtype=logits_fin.dtype
        )
        w_cap = pesos_clases["capitalizacion"].to(
            device=logits_cap.device, dtype=logits_cap.dtype
        )
    else:
        w_init = w_fin = w_cap = None

    # 2. Calculamos Cross-Entropy para cada tarea
    loss_init = F.cross_entropy(
        logits_init.reshape(-1, 2),
        y_init.reshape(-1),
        weight=w_init,
        ignore_index=-100
    )

    loss_fin = F.cross_entropy(
        logits_fin.reshape(-1, 4),
        y_fin.reshape(-1),
        weight=w_fin,
        ignore_index=-100
    )

    loss_cap = F.cross_entropy(
        logits_cap.reshape(-1, 4),
        y_cap.reshape(-1),
        weight=w_cap,
        ignore_index=-100
    )

    # 3. Mantenemos la misma importancia relativa entre tareas
    loss_total = (loss_init + loss_fin + loss_cap) / 3

    return loss_total, {
        "punt_inicial": loss_init,
        "punt_final": loss_fin,
        "capitalizacion": loss_cap
    }



def calcular_pesos_clases_rnn(secuencias_train, exponente=0.5, peso_maximo=5.0):
    """
    Calcula los pesos de las clases para Weighted Cross-Entropy
    utilizando exclusivamente el conjunto de entrenamiento.
    """

    esquema = [
        ("punt_inicial", "y_init", 2),
        ("punt_final", "y_fin", 4),
        ("capitalizacion", "y_cap", 4)
    ]

    pesos = {}

    for tarea, columna, n_clases in esquema:

        # 1. Recuperamos las etiquetas, excluyendo el padding (-100)
        etiquetas = secuencias_train[columna].reshape(-1)
        etiquetas = etiquetas[etiquetas != -100]

        # 2. Contamos cuántos tokens pertenecen a cada clase
        conteos = torch.bincount(etiquetas, minlength=n_clases)

        if conteos.numel() != n_clases or (conteos == 0).any():
            raise ValueError(
                f"Clases inválidas o ausentes en {tarea}: {conteos.tolist()}"
            )

        # 3. Calculamos pesos inversamente proporcionales
        # a la raíz cuadrada de las frecuencias
        pesos_tarea = (
            (conteos.max().float() / conteos.float())
            ** exponente
        ).clamp(max=peso_maximo)

        pesos[tarea] = pesos_tarea

        print(f"\n{tarea}")
        print("Cantidad de tokens:", conteos.tolist())
        print("Pesos:", pesos_tarea.tolist())

    return pesos
