import torch
import torch.nn as nn

class MultiHeadAttentionModel(nn.Module):
    def __init__(self, input_dim, action_dim, n_heads=4, hidden_dim=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        attn_output, _ = self.attention(x, x, x)
        out = attn_output[:, -1, :]  # son zaman adımı
        action = self.fc(out)
        return action 