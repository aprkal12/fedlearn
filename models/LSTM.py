from typing import Optional, Tuple

import torch
from torch import nn

class LSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] + [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        seq_len, batch_size = x.shape[:2] # x의 크기 : [seq_len, batch_size, input_size]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        out = []
        for t in range(seq_len):
            input = x[t]
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](input, h[layer], c[layer])
                input = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)
    
class LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias = False)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        i, f, g, o = ifgo.chunk(4, dim = -1) # ifgo 텐서를 4등분한다.
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)

        return h_next, c_next