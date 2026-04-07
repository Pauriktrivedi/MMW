import logging
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SampleStrategy(BaseStrategy):
    """
    A basic sample strategy that places a simple paper trade based on a dummy condition.
    Modify this class to implement your actual technical analysis and logic.
    """
    def __init__(self, mode='paper', paper_trader=None, live_trader=None, risk_manager=None):
        super().__init__(name="SampleStrategy", mode=mode, paper_trader=paper_trader, live_trader=live_trader, risk_manager=risk_manager)
        self.target_symbol = "nse_cm|26000" # NIFTY 50 Example Token
        self.trade_executed = False

    def on_tick(self, tick_data: dict):
        """
        Called every time a new market data tick arrives from the WebSocket.
        tick_data contains {'tk': token, 'e': exchange, 'ltp': last_traded_price, ...}
        """
        token = str(tick_data.get("tk", ""))
        exchange = tick_data.get("e", "")
        symbol = f"{exchange}|{token}"
        ltp = float(tick_data.get("ltp", 0.0))

        # Only process ticks for our target symbol
        if symbol == self.target_symbol:
            logger.info(f"[{self.name}] Tick received for {symbol}: LTP = {ltp}")

            # Example Logic: Buy 1 quantity if LTP is greater than 0 and we haven't traded yet today
            if ltp > 0 and not self.trade_executed:
                logger.info(f"[{self.name}] Trading condition met. Attempting to place order.")

                # Using dummy values for trading_symbol and exchange_seg
                trading_symbol = "NIFTY_EXAMPLE"
                qty = 1
                price = ltp

                order_id = self.place_order(
                    symbol=symbol,
                    trading_symbol=trading_symbol,
                    qty=qty,
                    side="BUY",
                    exchange_seg=exchange,
                    order_type="MARKET",
                    price=price,
                    instrument_type="EQ"
                )

                if order_id:
                    logger.info(f"[{self.name}] Order successfully placed with ID: {order_id}")
                    self.trade_executed = True
                else:
                    logger.warning(f"[{self.name}] Failed to place order or blocked by risk manager.")


    def on_signal(self, signal_data: dict):
        """
        Called when an external signal is generated (if using a separate signal generation engine).
        """
        pass

    def on_order_fill(self, trade_data: dict):
        """
        Called when an order is filled (either from live API webhook/poll or paper simulator).
        """
        logger.info(f"[{self.name}] Received fill confirmation: {trade_data}")
