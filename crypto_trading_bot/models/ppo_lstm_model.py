import torch
import torch.nn as nn

class PPO_LSTM_Model(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, dropout_p=0.2):
        super(PPO_LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)
        actor_out = self.actor(out)
        critic_out = self.critic(out)
        return actor_out, critic_out, hidden

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location)) 