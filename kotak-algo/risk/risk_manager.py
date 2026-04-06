import json
from datetime import date
import logging
from database.database import SessionLocal
from database.models import RiskState

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, max_daily_loss, per_trade_stop_loss, max_open_positions):
        self.max_daily_loss = float(max_daily_loss)
        self.per_trade_stop_loss = float(per_trade_stop_loss)
        self.max_open_positions = int(max_open_positions)
        self.today = date.today()
        self.daily_pnl = 0.0
        self.open_positions = {}
        self._load_state()

    def _load_state(self):
        db = SessionLocal()
        try:
            state = db.query(RiskState).filter(RiskState.date == self.today).first()
            if state:
                self.daily_pnl = state.daily_pnl
                self.open_positions = json.loads(state.open_positions_json)
            else:
                new_state = RiskState(
                    date=self.today,
                    daily_pnl=0.0,
                    open_positions_json="{}"
                )
                db.add(new_state)
                db.commit()
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")
        finally:
            db.close()

    def _save_state(self):
        db = SessionLocal()
        try:
            state = db.query(RiskState).filter(RiskState.date == self.today).first()
            if state:
                state.daily_pnl = self.daily_pnl
                state.open_positions_json = json.dumps(self.open_positions)
                db.commit()
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
        finally:
            db.close()

    def check_daily_loss_limit(self):
        return self.daily_pnl <= -self.max_daily_loss

    def check_max_positions(self):
        # Count non-zero positions
        active_positions = sum(1 for qty in self.open_positions.values() if qty != 0)
        return active_positions >= self.max_open_positions

    def check_order_allowed(self, symbol, side, qty, price):
        if self.check_daily_loss_limit():
            logger.warning(f"Order rejected: Daily loss limit exceeded ({self.daily_pnl}).")
            return False

        # Check if it's opening a new position or closing an existing one
        current_qty = self.open_positions.get(symbol, 0)
        is_closing = False

        if (side == "BUY" and current_qty < 0) or (side == "SELL" and current_qty > 0):
            is_closing = True

        if not is_closing and current_qty == 0 and self.check_max_positions():
            logger.warning(f"Order rejected: Max open positions reached ({self.max_open_positions}).")
            return False

        return True

    def calculate_position_size(self, symbol, current_price, account_balance):
        # Risk X% of account balance per trade
        # For simplicity, risking absolute per_trade_stop_loss amount
        risk_amount = self.per_trade_stop_loss
        # Need stop loss distance to calculate size. Assume 1% for this basic implementation if not provided
        stop_loss_pct = 0.01
        sl_distance = current_price * stop_loss_pct
        if sl_distance == 0:
            return 0

        qty = int(risk_amount / sl_distance)
        return qty

    def update_pnl(self, realized_pnl):
        self.daily_pnl += realized_pnl
        self._save_state()

    def update_position(self, symbol, qty, side):
        current_qty = self.open_positions.get(symbol, 0)
        if side == "BUY":
            current_qty += qty
        else:
            current_qty -= qty

        self.open_positions[symbol] = current_qty
        self._save_state()

    def update_unrealized_pnl(self, current_prices):
        """ Calculate unrealized MTM PnL across open positions based on average entry.
            Requires tracking average entry price which is omitted for brevity.
            Here we just return a stub sum or log. """
        pass
