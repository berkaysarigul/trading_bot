import unittest
from crypto_trading_bot.utils.risk_management import value_at_risk, estimate_slippage
from crypto_trading_bot.utils.performance_metrics import sharpe_ratio
import numpy as np

class TestUtils(unittest.TestCase):
    def test_value_at_risk(self):
        returns = np.random.normal(0, 1, 1000)
        var = value_at_risk(returns)
        self.assertTrue(var > 0)
    def test_slippage(self):
        slip = estimate_slippage(1000, 10000)
        self.assertTrue(0 <= slip <= 0.05)
    def test_sharpe(self):
        returns = np.random.normal(0, 1, 1000)
        s = sharpe_ratio(returns)
        self.assertIsInstance(s, float)

if __name__ == '__main__':
    unittest.main() 