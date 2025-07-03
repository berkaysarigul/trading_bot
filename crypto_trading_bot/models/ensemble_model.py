import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, models, action_dim):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fc = nn.Sequential(
            nn.Linear(len(models) * action_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Her modelin çıktısını birleştir
        outputs = [model(x)[0] if isinstance(model(x), tuple) else model(x) for model in self.models]
        concat = torch.cat(outputs, dim=-1)
        action = self.fc(concat)
        return action 