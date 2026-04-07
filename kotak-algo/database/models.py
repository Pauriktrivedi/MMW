from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)
    order_type = Column(String, nullable=False)  # MARKET/LIMIT/SL/SL-M
    side = Column(String, nullable=False)  # BUY/SELL
    status = Column(String, nullable=False)  # PENDING/FILLED/CANCELLED/REJECTED
    instrument_type = Column(String, nullable=False)  # CE/PE/FUT/EQ
    strike_price = Column(Float, nullable=True)
    expiry_date = Column(String, nullable=True)
    exchange_seg = Column(String, nullable=False)
    trading_symbol = Column(String, nullable=False)
    mode = Column(String, nullable=False)  # paper/live
    kotak_order_id = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_order_symbol_timestamp", "symbol", "timestamp"),
    )

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    side = Column(String, nullable=False)
    slippage = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    trading_symbol = Column(String, nullable=False)
    exchange_seg = Column(String, nullable=False)
    instrument_type = Column(String, nullable=False)
    bid_price = Column(Float, nullable=False)
    ask_price = Column(Float, nullable=False)
    last_traded_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    oi = Column(Integer, nullable=False) # open interest
    strike_price = Column(Float, nullable=True)
    expiry_date = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_marketdata_symbol_timestamp", "symbol", "timestamp"),
    )

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    signal_type = Column(String, nullable=False) # BUY/SELL
    strength = Column(Float, nullable=False)
    strategy_name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)

class PnlSummary(Base):
    __tablename__ = "pnl_summary"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False)
    total_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)

class RiskState(Base):
    __tablename__ = "risk_state"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    open_positions_json = Column(String, nullable=False) # stores dict as JSON
