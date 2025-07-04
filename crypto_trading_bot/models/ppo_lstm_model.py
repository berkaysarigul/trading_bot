import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class PPO_LSTM_Model(nn.Module):
    """
    PPO + LSTM model for crypto trading
    Stable-Baselines3 compatible
    """
    
    def __init__(self, input_dim, action_dim, hidden_dim=128, dropout_p=0.2, n_lstm_layers=2):
        super(PPO_LSTM_Model, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout_p if n_lstm_layers > 1 else 0
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            hidden: LSTM hidden state (h0, c0)
            
        Returns:
            actor_out: Action logits [batch_size, action_dim]
            critic_out: Value estimate [batch_size, 1]
            hidden: Updated LSTM hidden state
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Apply batch norm and dropout
        out = self.bn(out)
        out = self.dropout(out)
        
        # Actor and critic outputs
        actor_out = self.actor(out)
        critic_out = self.critic(out)
        
        return actor_out, critic_out, hidden
    
    def get_action(self, obs: np.ndarray, hidden: Tuple[torch.Tensor, torch.Tensor] = None, deterministic: bool = False) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from observation
        
        Args:
            obs: Observation array [sequence_length, input_dim]
            hidden: LSTM hidden state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action [action_dim]
            hidden: Updated LSTM hidden state
        """
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            actor_out, critic_out, hidden = self.forward(x, hidden)
            
            # Action selection
            if deterministic:
                action = torch.tanh(actor_out).squeeze(0).numpy()
            else:
                # Add noise for exploration
                noise = torch.randn_like(actor_out) * 0.1
                action = torch.tanh(actor_out + noise).squeeze(0).numpy()
            
            return action, hidden

    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'n_lstm_layers': self.n_lstm_layers
        }, path)

    def load(self, path: str, map_location: str = None):
        """Load model"""
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

# Stable-Baselines3 compatible policy
class MlpLstmPolicy:
    """
    Stable-Baselines3 compatible LSTM policy
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, lstm_hidden_size=128, n_lstm_layers=2, dropout=0.1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        
        # Create model
        input_dim = observation_space.shape[-1]  # Last dimension is features
        action_dim = action_space.shape[0]
        
        self.model = PPO_LSTM_Model(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=lstm_hidden_size,
            dropout_p=dropout,
            n_lstm_layers=n_lstm_layers
        )
    
    def forward(self, obs, deterministic=False):
        """Forward pass for Stable-Baselines3"""
        return self.model.get_action(obs, deterministic=deterministic) 