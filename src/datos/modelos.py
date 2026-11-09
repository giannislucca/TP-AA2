
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#comentario para ignronates como yo(gianis)
# "Head": es la V de la clase (o<t> = g2(V h<t>)), aplicada sobre el h<t> compartido
# que sale de la GRU/LSTM. En vez de una sola V que predice las C clases de UNA tarea
# (como en el ejemplo de la slide), acá hay 3 V's distintas -una por tarea-, cada una
# con su propio tamaño de salida, pero las tres leen el mismo h<t>.



# GRU multitarea para predicción de puntuación inicial, puntuación final y capitalización
class GRUMultitarea(nn.Module):
    """
    GRU unidireccional multitarea: a partir de embeddings de BERT por sub-token,
    predice punt_inicial, punt_final y capitalización para cada posición de la secuencia.
    """
    def __init__(self, input_size=768, hidden_size=128, dropout=0.2):
        
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
            bidirectional=False
        )

        # Dropout sobre la salida de la GRU, antes de los heads, para regularizar.
        self.dropout = nn.Dropout(dropout)

        # Una capa de salida por tarea
        # El número de salidas de cada head es la cantidad de clases de esa tarea:
        # punt_inicial tiene 2 clases, punt_final y capitalización tienen 4 cada una.
        self.head_init = nn.Linear(hidden_size, 2)
        self.head_fin = nn.Linear(hidden_size, 4)
        self.head_cap = nn.Linear(hidden_size, 4)

    def forward(self, x):
        # h_n es el último estado oculto (resumen de toda la secuencia). Al ser
        # m a m, nescesitamos todo el out, no solo el ultimo.
        out, h_n = self.gru(x)


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
    def __init__(self, input_size=768, hidden_size=128, dropout=0.2):

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
            bidirectional=False
        )

        # Dropout sobre la salida de la LSTM, antes de los heads, para regularizar.
        self.dropout = nn.Dropout(dropout)

        # Una capa de salida por tarea
        # El número de salidas de cada head es la cantidad de clases de esa tarea:
        # punt_inicial tiene 2 clases, punt_final y capitalización tienen 4 cada una.
        self.head_init = nn.Linear(hidden_size, 2)
        self.head_fin = nn.Linear(hidden_size, 4)
        self.head_cap = nn.Linear(hidden_size, 4)

    def forward(self, x):
        # La LSTM devuelve (h_n, c_n) en vez de un solo h_n: h_n es el último estado oculto
        # (igual que en la GRU) y c_n es el último cell state, la memoria de largo plazo que
        # arrastra aparte del hidden state. Al ser m a m, nescesitamos todo el out, no h_n ni c_n.
        out, (h_n, c_n) = self.lstm(x)


        out = self.dropout(out)# esto es h<t> para todo t

        logits_init = self.head_init(out) # o<t> para la tarea 1, vía su propia V (tamaño 2) (sin softmax aplicado, para las de abajo tmb)
        logits_fin = self.head_fin(out) # o<t> para la tarea 2, vía su propia V (tamaño 4)
        logits_cap = self.head_cap(out) # o<t> para la tarea 3, vía su propia V (tamaño 4)

        return logits_init, logits_fin, logits_cap

# F de perdida

def calcular_perdida_multitarea(logits_init, logits_fin, logits_cap,
                               y_init, y_fin, y_cap):
    """
    Calcula la pérdida conjunta de las tres tareas.
    Ignora las posiciones de padding (etiqueta -100).
    """

    loss_init = F.cross_entropy(
        logits_init.reshape(-1, 2),
        y_init.reshape(-1),
        ignore_index=-100
    )

    loss_fin = F.cross_entropy(
        logits_fin.reshape(-1, 4),
        y_fin.reshape(-1),
        ignore_index=-100
    )

    loss_cap = F.cross_entropy(
        logits_cap.reshape(-1, 4),
        y_cap.reshape(-1),
        ignore_index=-100
    )

    loss_total = (loss_init + loss_fin + loss_cap) / 3

    return loss_total, {
        "punt_inicial": loss_init,
        "punt_final": loss_fin,
        "capitalizacion": loss_cap
    }

