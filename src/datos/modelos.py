
import torch
import torch.nn as nn
import torch.nn.functional as F

# GRU multitarea para predicción de puntuación inicial, puntuación final y capitalización
class GRUMultitarea(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        # Una capa de salida por tarea
        self.head_init = nn.Linear(hidden_size, 2)
        self.head_fin = nn.Linear(hidden_size, 4)
        self.head_cap = nn.Linear(hidden_size, 4)

    def forward(self, x):
        # x: (B, T, 768)
        out, h_n = self.gru(x)

        # out: (B, T, hidden_size)
        out = self.dropout(out)

        logits_init = self.head_init(out)
        logits_fin = self.head_fin(out)
        logits_cap = self.head_cap(out)

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

