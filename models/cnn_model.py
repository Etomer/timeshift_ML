import torch
import torch.nn as nn


# class Block(nn.Module):
#     def __init__(self,size, dropout):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.l = nn.Linear(size,2*size)
#         self.l2 = nn.Linear(2*size,size)
#         self.act = nn.GELU()
#         self.ln = nn.LayerNorm(size)
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)

#     def forward(self, x):
#         return x + self.l2(self.act(self.l(self.ln(self.dropout(x)))))


class model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.thinker = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config["dropout"]),
            nn.Linear(
                config["scale_factor"] * config["cnn_output_size_at_factor_1"], 1000
            ),
            nn.GELU(),
            # Block(1000,dropout),
            # Block(1000,dropout),
            nn.Linear(1000, config["guess_grid_size"]),
        )

        self.apply(self._init_weights)

        self.cnn = nn.Sequential(
            nn.Conv1d(4, 48 * config["scale_factor"], 50, stride=5),
            nn.GELU(),
            nn.Conv1d(
                48 * config["scale_factor"], 48 * config["scale_factor"], 50, stride=5
            ),
            nn.GELU(),
            nn.Conv1d(
                48 * config["scale_factor"], 48 * config["scale_factor"], 30, stride=5
            ),
            nn.GELU(),
            nn.Flatten(),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.00002)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.00002)

    def forward(self, x):
        x = self.cnn(x)
        x = self.thinker(x)
        return x
