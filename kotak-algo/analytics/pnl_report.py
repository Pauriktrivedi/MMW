import logging
from datetime import date
from database.database import SessionLocal
from database.models import PnlSummary, Trade

logger = logging.getLogger(__name__)

class PnlReport:
    def __init__(self):
        pass

    def calculate_daily_pnl(self, run_date=None):
        if not run_date:
            run_date = date.today()

        db = SessionLocal()
        try:
            # Get all trades for the given date
            # Ensure we are filtering by date correctly. Depending on DB dialect,
            # cast to date or filter by range. Here we do simple range.
            from datetime import datetime, time
            start_dt = datetime.combine(run_date, time.min)
            end_dt = datetime.combine(run_date, time.max)

            trades = db.query(Trade).filter(Trade.timestamp >= start_dt, Trade.timestamp <= end_dt).all()

            realized_pnl = 0.0
            total_trades = len(trades)
            winning_trades = 0

            # Group trades by symbol to match buys and sells
            symbol_trades = {}
            for t in trades:
                if t.symbol not in symbol_trades:
                    symbol_trades[t.symbol] = []
                symbol_trades[t.symbol].append(t)

            for symbol, s_trades in symbol_trades.items():
                # Sort by time
                s_trades.sort(key=lambda x: x.timestamp)

                # Simple FIFO matching for realized PnL
                buy_queue = []
                sell_queue = []

                for t in s_trades:
                    qty = t.quantity
                    price = t.price

                    if t.side == "BUY":
                        while qty > 0 and sell_queue:
                            match = sell_queue[0]
                            match_qty = min(qty, match["qty"])
                            # We bought to close a short
                            pnl = (match["price"] - price) * match_qty
                            realized_pnl += pnl
                            if pnl > 0:
                                winning_trades += 1

                            qty -= match_qty
                            match["qty"] -= match_qty
                            if match["qty"] == 0:
                                sell_queue.pop(0)

                        if qty > 0:
                            buy_queue.append({"qty": qty, "price": price})

                    elif t.side == "SELL":
                        while qty > 0 and buy_queue:
                            match = buy_queue[0]
                            match_qty = min(qty, match["qty"])
                            # We sold to close a long
                            pnl = (price - match["price"]) * match_qty
                            realized_pnl += pnl
                            if pnl > 0:
                                winning_trades += 1

                            qty -= match_qty
                            match["qty"] -= match_qty
                            if match["qty"] == 0:
                                buy_queue.pop(0)

                        if qty > 0:
                            sell_queue.append({"qty": qty, "price": price})

            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

            # Store it
            summary = db.query(PnlSummary).filter(PnlSummary.date == run_date).first()
            if not summary:
                summary = PnlSummary(
                    date=run_date,
                    total_pnl=realized_pnl,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=0.0,
                    win_rate=win_rate,
                    total_trades=total_trades,
                    winning_trades=winning_trades
                )
                db.add(summary)
            else:
                summary.total_pnl = realized_pnl
                summary.realized_pnl = realized_pnl
                summary.win_rate = win_rate
                summary.total_trades = total_trades
                summary.winning_trades = winning_trades

            db.commit()
            return summary
        except Exception as e:
            logger.error(f"Error calculating daily PnL: {e}")
        finally:
            db.close()

    def get_today_pnl(self):
        db = SessionLocal()
        try:
            summary = db.query(PnlSummary).filter(PnlSummary.date == date.today()).first()
            if summary:
                return summary.total_pnl
            return 0.0
        finally:
            db.close()
