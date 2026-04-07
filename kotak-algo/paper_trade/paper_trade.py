import os
import time
import random
import threading
import logging
from datetime import datetime
from dotenv import load_dotenv
from database.database import SessionLocal
from database.models import Order, Trade, MarketData

logger = logging.getLogger(__name__)

class PaperTradeSimulator:
    def __init__(self):
        load_dotenv()
        self.virtual_cash = float(os.getenv("VIRTUAL_CASH", "500000"))
        self.starting_cash = self.virtual_cash
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._simulate_fills_loop, daemon=True)
        self.thread.start()
        logger.info("Paper Trade Simulator started in background.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Paper Trade Simulator stopped.")

    def place_order(self, symbol, trading_symbol, qty, side, exchange_seg, order_type, price, instrument_type, strike_price, expiry_date):
        db = SessionLocal()
        try:
            new_order = Order(
                symbol=symbol,
                quantity=int(qty),
                price=float(price) if price != '0' and price else None,
                order_type=order_type,
                side=side.upper(),
                status="PENDING",
                instrument_type=instrument_type,
                strike_price=strike_price,
                expiry_date=expiry_date,
                exchange_seg=exchange_seg,
                trading_symbol=trading_symbol,
                mode="paper",
                kotak_order_id=f"PAPER_{int(time.time()*1000)}",
                timestamp=datetime.now()
            )
            db.add(new_order)
            db.commit()
            logger.info(f"Placed PAPER order {new_order.kotak_order_id} for {side} {qty} {trading_symbol}")
            return new_order.kotak_order_id
        except Exception as e:
            logger.error(f"Failed to place paper order: {e}")
            raise
        finally:
            db.close()

    def _simulate_fills_loop(self):
        # We also generate mock ticks for the paper dashboard to simulate a live market
        tick_counter = 0
        while self.running:
            self._simulate_fills()
            if tick_counter % 5 == 0:
                self._generate_mock_ticks()
            tick_counter += 1
            time.sleep(1.0)

    def _generate_mock_ticks(self):
        db = SessionLocal()
        try:
            # Fetch latest tick for each symbol
            from sqlalchemy import func
            subquery = db.query(
                MarketData.symbol,
                func.max(MarketData.timestamp).label("max_timestamp")
            ).group_by(MarketData.symbol).subquery()

            latest_data = db.query(MarketData).join(
                subquery,
                (MarketData.symbol == subquery.c.symbol) &
                (MarketData.timestamp == subquery.c.max_timestamp)
            ).all()

            now = datetime.now()
            for md in latest_data:
                # Random walk
                change_pct = random.uniform(-0.001, 0.001)
                new_ltp = md.last_traded_price * (1 + change_pct)
                new_bid = new_ltp * 0.999
                new_ask = new_ltp * 1.001

                new_md = MarketData(
                    symbol=md.symbol,
                    trading_symbol=md.trading_symbol,
                    exchange_seg=md.exchange_seg,
                    instrument_type=md.instrument_type,
                    bid_price=new_bid,
                    ask_price=new_ask,
                    last_traded_price=new_ltp,
                    volume=md.volume + random.randint(10, 100),
                    oi=md.oi + random.randint(-10, 10),
                    strike_price=md.strike_price,
                    expiry_date=md.expiry_date,
                    timestamp=now
                )
                db.add(new_md)
            db.commit()
        except Exception as e:
            logger.error(f"Error generating mock ticks: {e}")
        finally:
            db.close()

    def _simulate_fills(self):
        db = SessionLocal()
        try:
            pending_orders = db.query(Order).filter(Order.status == "PENDING", Order.mode == "paper").all()
            for order in pending_orders:
                # Fetch latest market data for the symbol
                latest_md = db.query(MarketData).filter(MarketData.symbol == order.symbol).order_by(MarketData.timestamp.desc()).first()
                if not latest_md:
                    continue

                filled = False
                fill_price = 0.0
                slippage = 0.0

                if order.order_type == "MARKET":
                    filled = True
                    if order.side == "BUY":
                        slippage_pct = random.uniform(0.0001, 0.0005)
                        fill_price = latest_md.ask_price * (1 + slippage_pct)
                    else:
                        slippage_pct = random.uniform(0.0001, 0.0005)
                        fill_price = latest_md.bid_price * (1 - slippage_pct)
                    slippage = abs(fill_price - (latest_md.ask_price if order.side == "BUY" else latest_md.bid_price))

                elif order.order_type == "LIMIT":
                    if order.side == "BUY" and latest_md.last_traded_price <= order.price:
                        filled = True
                        fill_price = order.price
                    elif order.side == "SELL" and latest_md.last_traded_price >= order.price:
                        filled = True
                        fill_price = order.price

                if filled:
                    order.status = "FILLED"

                    new_trade = Trade(
                        order_id=order.id,
                        symbol=order.symbol,
                        quantity=order.quantity,
                        price=fill_price,
                        side=order.side,
                        slippage=slippage,
                        timestamp=datetime.now()
                    )
                    db.add(new_trade)

                    # Update virtual cash roughly (assumes fully cash settled without margin for simplicity)
                    trade_val = order.quantity * fill_price
                    if order.side == "BUY":
                        self.virtual_cash -= trade_val
                    else:
                        self.virtual_cash += trade_val

                    db.commit()
                    logger.info(f"PAPER Fill: {order.side} {order.quantity} {order.symbol} @ {fill_price:.2f}")

        except Exception as e:
            logger.error(f"Error in paper trade simulation loop: {e}")
            db.rollback()
        finally:
            db.close()

    def get_virtual_pnl(self):
        return {
            "starting_cash": self.starting_cash,
            "current_cash": self.virtual_cash,
            "cash_pnl": self.virtual_cash - self.starting_cash
        }

    def get_open_positions(self):
        db = SessionLocal()
        try:
            trades = db.query(Trade).join(Order).filter(Order.mode == "paper").all()
            positions = {}
            for t in trades:
                if t.symbol not in positions:
                    positions[t.symbol] = 0
                if t.side == "BUY":
                    positions[t.symbol] += t.quantity
                else:
                    positions[t.symbol] -= t.quantity

            # Remove closed positions
            return {sym: qty for sym, qty in positions.items() if qty != 0}
        finally:
            db.close()
