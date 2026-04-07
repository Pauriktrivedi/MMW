import unittest
import os
import time
from datetime import datetime
from database.database import init_db, SessionLocal, engine
from database.models import Base, Order, Trade, MarketData, RiskState
from risk.risk_manager import RiskManager
from paper_trade.paper_trade import PaperTradeSimulator
from strategies.base_strategy import BaseStrategy

# A dummy strategy for testing instantiation in paper mode
class DummyStrategy(BaseStrategy):
    def __init__(self, mode='paper', paper_trader=None, risk_manager=None):
        super().__init__("Dummy", mode=mode, paper_trader=paper_trader, risk_manager=risk_manager)

    def on_tick(self, tick_data):
        pass
    def on_signal(self, signal):
        pass
    def on_order_fill(self, trade):
        pass


class TestTradingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure we are using a test database by overriding URL if needed
        # For this test, we just drop and recreate the sqlite db tables
        os.environ["PAPER_MODE"] = "true"
        os.environ["VIRTUAL_CASH"] = "500000"
        os.environ["MAX_DAILY_LOSS"] = "1000"
        os.environ["MAX_OPEN_POSITIONS"] = "2"
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        # Clean db before each test
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def test_database_tables_created(self):
        db = SessionLocal()
        try:
            # Insert a dummy record
            md = MarketData(
                symbol="nse_cm|1234",
                trading_symbol="RELIANCE",
                exchange_seg="nse_cm",
                instrument_type="EQ",
                bid_price=100.0,
                ask_price=101.0,
                last_traded_price=100.5,
                volume=1000,
                oi=0,
                timestamp=datetime.now()
            )
            db.add(md)
            db.commit()

            count = db.query(MarketData).count()
            self.assertEqual(count, 1)
        finally:
            db.close()

    def test_risk_manager_limits(self):
        rm = RiskManager(max_daily_loss=1000, per_trade_stop_loss=100, max_open_positions=2)

        # Test max positions
        rm.open_positions = {"TCS": 1, "INFY": 1}
        self.assertFalse(rm.check_order_allowed("WIPRO", "BUY", 1, 100))

        # Closing is allowed
        self.assertTrue(rm.check_order_allowed("TCS", "SELL", 1, 100))

        # Test daily loss limit
        rm.open_positions = {}
        rm.daily_pnl = -1500
        self.assertFalse(rm.check_order_allowed("TCS", "BUY", 1, 100))

    def test_paper_trade_fill_cycle(self):
        db = SessionLocal()
        try:
            # 1. Insert Market Data
            md = MarketData(
                symbol="nse_cm|1234",
                trading_symbol="RELIANCE",
                exchange_seg="nse_cm",
                instrument_type="EQ",
                bid_price=2500.0,
                ask_price=2501.0,
                last_traded_price=2500.5,
                volume=1000,
                oi=0,
                timestamp=datetime.now()
            )
            db.add(md)
            db.commit()

            # 2. Start Paper Trader
            pt = PaperTradeSimulator()
            pt.start()

            # 3. Place Order
            order_id = pt.place_order(
                symbol="nse_cm|1234",
                trading_symbol="RELIANCE",
                qty=10,
                side="BUY",
                exchange_seg="nse_cm",
                order_type="MARKET",
                price='0',
                instrument_type="EQ",
                strike_price=None,
                expiry_date=None
            )

            # 4. Wait for fill cycle
            time.sleep(2)
            pt.stop()

            # 5. Assert Fill
            order = db.query(Order).filter(Order.kotak_order_id == order_id).first()
            self.assertIsNotNone(order)
            self.assertEqual(order.status, "FILLED")

            trade = db.query(Trade).filter(Trade.order_id == order.id).first()
            self.assertIsNotNone(trade)
            self.assertEqual(trade.side, "BUY")
            self.assertEqual(trade.quantity, 10)
            # Price should be roughly ask_price (2501) with some slippage
            self.assertGreaterEqual(trade.price, 2501.0)

        finally:
            db.close()

    def test_strategy_instantiation(self):
        pt = PaperTradeSimulator()
        rm = RiskManager(1000, 100, 2)
        strat = DummyStrategy(mode='paper', paper_trader=pt, risk_manager=rm)

        self.assertEqual(strat.mode, 'paper')
        self.assertIsNotNone(strat.paper_trader)
        self.assertIsNotNone(strat.risk_manager)


if __name__ == '__main__':
    unittest.main()
