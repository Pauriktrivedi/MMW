from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, name, mode='paper', paper_trader=None, live_trader=None, risk_manager=None):
        self.name = name
        self.mode = mode
        self.paper_trader = paper_trader
        self.live_trader = live_trader
        self.risk_manager = risk_manager

    @abstractmethod
    def on_tick(self, tick_data: dict):
        pass

    @abstractmethod
    def on_signal(self, signal_data: dict):
        pass

    @abstractmethod
    def on_order_fill(self, trade_data: dict):
        pass

    def place_order(self, symbol, trading_symbol, qty, side, exchange_seg, order_type='MARKET', price='0', instrument_type='EQ', strike_price=None, expiry_date=None):

        # Risk check
        if self.risk_manager:
            if not self.risk_manager.check_order_allowed(symbol, side, qty, price):
                logger.info(f"Order blocked by risk manager for {symbol}")
                return None

        # Route based on mode
        if self.mode == 'paper':
            if self.paper_trader:
                return self.paper_trader.place_order(symbol, trading_symbol, qty, side, exchange_seg, order_type, price, instrument_type, strike_price, expiry_date)
            else:
                logger.error("Paper trader not initialized.")
        elif self.mode == 'live':
            if self.live_trader:
                return self.live_trader.place_order(symbol, trading_symbol, qty, side, exchange_seg, order_type, price, "NRML", '0', instrument_type, strike_price, expiry_date)
            else:
                logger.error("Live trader not initialized.")
        else:
            logger.error(f"Unknown trading mode: {self.mode}")

        return None
