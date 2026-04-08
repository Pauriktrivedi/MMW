import logging
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutRangeStrategy(BaseStrategy):
    def __init__(self, mode='paper', paper_trader=None, live_trader=None, risk_manager=None, symbol=None, range_high=None, range_low=None, quantity=100):
        super().__init__(name="BreakoutRange", mode=mode, paper_trader=paper_trader, live_trader=live_trader, risk_manager=risk_manager)
        self.symbol = symbol or "nse_cm|Nifty 50" # Default testing symbol
        self.range_high = range_high
        self.range_low = range_low
        self.quantity = quantity
        self.position = 0 # 0 for flat, 1 for long, -1 for short
        self.has_traded = False

        if not self.range_high or not self.range_low:
             # Example range for Nifty 50. In reality, you'd calculate this from history.
             self.range_high = 22050.0
             self.range_low = 21950.0

        logger.info(f"Initialized BreakoutRangeStrategy for {self.symbol}. Range: [{self.range_low}, {self.range_high}]")

    def on_tick(self, tick_data: dict):
        if self.has_traded:
            return # Only take one trade per day for this simple example

        symbol = tick_data.get('symbol')
        if symbol != self.symbol:
            return

        ltp = tick_data.get('last_traded_price')
        if not ltp:
            return

        logger.debug(f"{self.symbol} LTP: {ltp} | Range: [{self.range_low}, {self.range_high}]")

        # Breakout to the upside
        if ltp > self.range_high and self.position == 0:
            logger.info(f"Upside breakout detected for {self.symbol} at {ltp}. Placing BUY order.")
            order_id = self.place_order(
                symbol=self.symbol,
                trading_symbol=tick_data.get("trading_symbol", self.symbol),
                qty=self.quantity,
                side="BUY",
                exchange_seg=tick_data.get("exchange_seg", "nse_cm"),
                order_type="MARKET",
                price=0,
                instrument_type=tick_data.get("instrument_type", "EQ")
            )
            if order_id:
                self.position = 1
                self.has_traded = True

        # Breakout to the downside
        elif ltp < self.range_low and self.position == 0:
            logger.info(f"Downside breakout detected for {self.symbol} at {ltp}. Placing SELL order.")
            order_id = self.place_order(
                symbol=self.symbol,
                trading_symbol=tick_data.get("trading_symbol", self.symbol),
                qty=self.quantity,
                side="SELL",
                exchange_seg=tick_data.get("exchange_seg", "nse_cm"),
                order_type="MARKET",
                price=0,
                instrument_type=tick_data.get("instrument_type", "EQ")
            )
            if order_id:
                self.position = -1
                self.has_traded = True

    def on_signal(self, signal_data: dict):
        pass

    def on_order_fill(self, trade_data: dict):
        logger.info(f"Breakout strategy trade filled: {trade_data}")
