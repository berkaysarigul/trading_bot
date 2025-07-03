import unittest
from crypto_trading_bot.environments.crypto_env import CryptoTradingEnv

class DummyData:
    def __getitem__(self, key):
        return {'1h': __import__('pandas').DataFrame({'close':[1,2,3,4,5,6,7,8,9,10],'volume':[100]*10})}
    def keys(self):
        return ['BTC/USDT']

class TestEnv(unittest.TestCase):
    def test_env_reset_and_step(self):
        data = DummyData()
        env = CryptoTradingEnv(data)
        obs = env.reset()
        self.assertIsNotNone(obs)
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        self.assertIsInstance(reward, float)

if __name__ == '__main__':
    unittest.main() 