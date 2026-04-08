import logging
import pandas as pd
from datetime import datetime, time
import random
from database.database import SessionLocal
from database.models import Order, Trade

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy, historical_data_csv, initial_capital=500000):
        self.strategy = strategy
        self.historical_data_csv = historical_data_csv
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions = {}  # {symbol: {'qty': X, 'entry_price': Y, 'type': 'CE'/'PE'}}
        self.orders = []
        self.trades = []
        self.current_time = None
        self.pnl_history = []
        self.tick_data = None
        self.last_ltp = {} # {symbol: ltp}

    def load_data(self):
        logger.info(f"Loading historical data from {self.historical_data_csv}...")
        try:
            self.tick_data = pd.read_csv(self.historical_data_csv)
            # Ensure 'timestamp' column is parsed correctly to datetime
            if 'timestamp' in self.tick_data.columns:
                self.tick_data['timestamp'] = pd.to_datetime(self.tick_data['timestamp'])

            # Sort by timestamp to ensure chronological replay
            self.tick_data = self.tick_data.sort_values(by='timestamp')
            logger.info(f"Loaded {len(self.tick_data)} rows.")
            return True
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return False

    def place_order(self, symbol, trading_symbol, qty, side, exchange_seg, order_type='MARKET', price='0', instrument_type='EQ', strike_price=None, expiry_date=None):
        # Simulate order execution immediately for MARKET, or log it

        # Get the latest price for this symbol
        current_price = self.last_ltp.get(symbol, float(price) if price != '0' else 0)

        if current_price == 0:
            logger.warning(f"Cannot place order for {symbol}: No price data available.")
            return None

        # Simulate slippage for MARKET orders
        fill_price = current_price
        if order_type == 'MARKET':
            slippage = current_price * random.uniform(0.0001, 0.0005)
            fill_price = current_price + slippage if side == 'BUY' else current_price - slippage

        order_id = f"BT_ORD_{len(self.orders) + 1}"
        trade_id = f"BT_TRD_{len(self.trades) + 1}"

        order = {
            'id': order_id,
            'symbol': symbol,
            'trading_symbol': trading_symbol,
            'quantity': qty,
            'price': float(price),
            'order_type': order_type,
            'side': side,
            'status': 'FILLED',
            'timestamp': self.current_time
        }
        self.orders.append(order)

        trade = {
            'id': trade_id,
            'order_id': order_id,
            'symbol': symbol,
            'quantity': qty,
            'price': fill_price,
            'side': side,
            'timestamp': self.current_time
        }
        self.trades.append(trade)

        # Update portfolio
        if side == 'BUY':
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {'qty': 0, 'entry_price': 0, 'type': instrument_type}

            total_cost = fill_price * qty
            new_qty = self.open_positions[symbol]['qty'] + qty

            # Weighted average price
            if self.open_positions[symbol]['qty'] == 0:
                self.open_positions[symbol]['entry_price'] = fill_price
            else:
                old_value = self.open_positions[symbol]['qty'] * self.open_positions[symbol]['entry_price']
                self.open_positions[symbol]['entry_price'] = (old_value + total_cost) / new_qty

            self.open_positions[symbol]['qty'] = new_qty
            self.current_capital -= total_cost
        else: # SELL
            if symbol not in self.open_positions:
                self.open_positions[symbol] = {'qty': 0, 'entry_price': 0, 'type': instrument_type}

            total_revenue = fill_price * qty
            new_qty = self.open_positions[symbol]['qty'] - qty

            # Weighted average price for shorts
            if self.open_positions[symbol]['qty'] == 0:
                 self.open_positions[symbol]['entry_price'] = fill_price
            elif self.open_positions[symbol]['qty'] > 0:
                # Covering a long
                realized_pnl = (fill_price - self.open_positions[symbol]['entry_price']) * qty
                self.current_capital += realized_pnl
            else:
                 old_value = abs(self.open_positions[symbol]['qty']) * self.open_positions[symbol]['entry_price']
                 self.open_positions[symbol]['entry_price'] = (old_value + total_revenue) / abs(new_qty)

            self.open_positions[symbol]['qty'] = new_qty
            self.current_capital += total_revenue

        logger.info(f"[{self.current_time}] BT Order Filled: {side} {qty} {symbol} @ {fill_price:.2f}")

        # Inform strategy
        if hasattr(self.strategy, 'on_order_fill'):
            self.strategy.on_order_fill(trade)

        return order_id

    def calculate_unrealized_pnl(self):
        unrealized = 0
        for symbol, pos in self.open_positions.items():
            if pos['qty'] == 0:
                continue

            current_price = self.last_ltp.get(symbol, pos['entry_price'])
            if pos['qty'] > 0:
                unrealized += (current_price - pos['entry_price']) * pos['qty']
            else:
                unrealized += (pos['entry_price'] - current_price) * abs(pos['qty'])
        return unrealized

    def run(self):
        if self.tick_data is None:
            if not self.load_data():
                return

        logger.info("Starting backtest replay...")

        # Override the strategy's place_order to route to our backtester
        original_place_order = self.strategy.place_order
        self.strategy.place_order = self.place_order
        self.strategy.mode = 'backtest'

        # Replay ticks
        for index, row in self.tick_data.iterrows():
            self.current_time = row['timestamp']

            # Map row to standard tick dict format
            tick = row.to_dict()

            # Keep track of latest prices
            symbol = tick.get('symbol')
            if symbol and 'last_traded_price' in tick:
                self.last_ltp[symbol] = tick['last_traded_price']

            # Pass to strategy
            self.strategy.on_tick(tick)

            # Track PnL over time
            if index % 100 == 0:
                unrealized = self.calculate_unrealized_pnl()
                self.pnl_history.append({
                    'timestamp': self.current_time,
                    'capital': self.current_capital,
                    'unrealized': unrealized,
                    'total': self.current_capital + unrealized
                })

        logger.info("Backtest complete.")
        self.report()

        # Restore strategy method
        self.strategy.place_order = original_place_order

    def report(self):
        unrealized = self.calculate_unrealized_pnl()
        final_capital = self.current_capital + unrealized
        total_pnl = final_capital - self.initial_capital

        print("\n" + "="*40)
        print("BACKTEST REPORT")
        print("="*40)
        print(f"Initial Capital: {self.initial_capital:.2f}")
        print(f"Final Capital:   {final_capital:.2f}")
        print(f"Total PnL:       {total_pnl:.2f} ({(total_pnl/self.initial_capital)*100:.2f}%)")
        print(f"Total Trades:    {len(self.trades)}")
        print("="*40)
