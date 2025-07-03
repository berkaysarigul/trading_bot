import unittest
import torch
from crypto_trading_bot.models.ppo_lstm_model import PPO_LSTM_Model

class TestModel(unittest.TestCase):
    def test_forward(self):
        model = PPO_LSTM_Model(input_dim=4, action_dim=2)
        x = torch.randn(1, 5, 4)  # (batch, seq, input_dim)
        actor_out, critic_out, _ = model(x)
        self.assertEqual(actor_out.shape[-1], 2)
        self.assertEqual(critic_out.shape[-1], 1)

if __name__ == '__main__':
    unittest.main() 