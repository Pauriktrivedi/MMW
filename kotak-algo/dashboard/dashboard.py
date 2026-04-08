from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from dotenv import load_dotenv
from database.database import SessionLocal
from database.models import Order, Trade, PnlSummary, MarketData
from datetime import date
from sqlalchemy import func

app = FastAPI()

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Kotak Algo Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
        h1 { color: #333; }
        .header { display: flex; justify-content: space-between; align-items: center; background: #fff; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .badge { padding: 5px 10px; border-radius: 5px; color: white; font-weight: bold; }
        .paper { background-color: #ff9800; }
        .live { background-color: #f44336; }
        .status-dot { height: 15px; width: 15px; background-color: #ccc; border-radius: 50%; display: inline-block; margin-right: 10px; }
        .status-dot.connected { background-color: #4caf50; box-shadow: 0 0 8px #4caf50; }
        .status-dot.disconnected { background-color: #f44336; box-shadow: 0 0 8px #f44336; }
        .header-controls { display: flex; align-items: center; }
        .card { background: #fff; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .pnl { font-size: 2em; font-weight: bold; }
        .positive { color: #4caf50; }
        .negative { color: #f44336; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Kotak Algo System</h1>
        <div class="header-controls">
            <span id="ws-status-dot" class="status-dot disconnected" title="Disconnected"></span>
            <span id="mode-badge" class="badge">Loading...</span>
        </div>
    </div>

    <div class="card">
        <h2>Today's P&L</h2>
        <div id="pnl-amount" class="pnl">₹0.00</div>
    </div>

    <div class="card">
        <h2>Live Indices</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>LTP</th>
                    <th>Bid</th>
                    <th>Ask</th>
                    <th>Volume</th>
                </tr>
            </thead>
            <tbody id="indices-body">
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Option Chain (Live)</h2>
        <table>
            <thead>
                <tr>
                    <th>Call Bid</th>
                    <th>Call Ask</th>
                    <th>Call LTP</th>
                    <th>Call OI</th>
                    <th>Strike</th>
                    <th>Put LTP</th>
                    <th>Put Bid</th>
                    <th>Put Ask</th>
                    <th>Put OI</th>
                </tr>
            </thead>
            <tbody id="options-body">
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Open Positions</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                </tr>
            </thead>
            <tbody id="positions-body">
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Recent Trades</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Price</th>
                </tr>
            </thead>
            <tbody id="trades-body">
            </tbody>
        </table>
    </div>

    <script>
        async function fetchData() {
            try {
                const statusRes = await fetch('/api/status');
                const statusData = await statusRes.json();
                const badge = document.getElementById('mode-badge');
                if (statusData.mode === 'paper') {
                    badge.textContent = 'PAPER MODE';
                    badge.className = 'badge paper';
                } else {
                    badge.textContent = 'LIVE MODE';
                    badge.className = 'badge live';
                }

                const dot = document.getElementById('ws-status-dot');
                if (statusData.ws_connected) {
                    dot.className = 'status-dot connected';
                    dot.title = 'Connected';
                } else {
                    dot.className = 'status-dot disconnected';
                    dot.title = 'Disconnected';
                }

                const pnlRes = await fetch('/api/pnl');
                const pnlData = await pnlRes.json();
                const pnlEl = document.getElementById('pnl-amount');
                pnlEl.textContent = `₹${pnlData.pnl.toFixed(2)}`;
                pnlEl.className = `pnl ${pnlData.pnl >= 0 ? 'positive' : 'negative'}`;

                const tradesRes = await fetch('/api/trades');
                const tradesData = await tradesRes.json();
                const tradesBody = document.getElementById('trades-body');
                tradesBody.innerHTML = '';
                tradesData.forEach(t => {
                    tradesBody.innerHTML += `<tr>
                        <td>${new Date(t.timestamp).toLocaleTimeString()}</td>
                        <td>${t.symbol}</td>
                        <td>${t.side}</td>
                        <td>${t.quantity}</td>
                        <td>${t.price}</td>
                    </tr>`;
                });

                const posRes = await fetch('/api/positions');
                const posData = await posRes.json();
                const posBody = document.getElementById('positions-body');
                posBody.innerHTML = '';
                for (const [sym, qty] of Object.entries(posData)) {
                    posBody.innerHTML += `<tr>
                        <td>${sym}</td>
                        <td>${qty}</td>
                    </tr>`;
                }

                // Fetch live market data
                const mdRes = await fetch('/api/market_data');
                const mdData = await mdRes.json();

                const indicesBody = document.getElementById('indices-body');
                const optionsBody = document.getElementById('options-body');
                indicesBody.innerHTML = '';
                optionsBody.innerHTML = '';

                const optionsMap = {};

                mdData.forEach(item => {
                    // Check if it is an option (either by instrument type or having a strike price)
                    if (item.instrument_type === 'CE' || item.instrument_type === 'PE' || item.strike_price) {
                        const strike = item.strike_price;
                        if (!optionsMap[strike]) {
                            optionsMap[strike] = { CE: null, PE: null };
                        }
                        optionsMap[strike][item.instrument_type === 'CE' ? 'CE' : 'PE'] = item;
                    } else {
                        // Treat as index/equity
                        indicesBody.innerHTML += `<tr>
                            <td>${item.trading_symbol || item.symbol}</td>
                            <td>${item.ltp}</td>
                            <td>${item.bid}</td>
                            <td>${item.ask}</td>
                            <td>${item.volume}</td>
                        </tr>`;
                    }
                });

                // Render options chain sorted by strike
                const strikes = Object.keys(optionsMap).map(Number).sort((a,b) => a-b);
                strikes.forEach(strike => {
                    const ce = optionsMap[strike].CE || {};
                    const pe = optionsMap[strike].PE || {};
                    optionsBody.innerHTML += `<tr>
                        <td>${ce.bid || '-'}</td>
                        <td>${ce.ask || '-'}</td>
                        <td>${ce.ltp || '-'}</td>
                        <td>${ce.oi || '-'}</td>
                        <td style="font-weight: bold; text-align: center; background: #f0f0f0;">${strike}</td>
                        <td>${pe.ltp || '-'}</td>
                        <td>${pe.bid || '-'}</td>
                        <td>${pe.ask || '-'}</td>
                        <td>${pe.oi || '-'}</td>
                    </tr>`;
                });

            } catch (e) {
                console.error("Error fetching data", e);
            }
        }

        setInterval(fetchData, 5000);
        fetchData();
    </script>
</body>
</html>
"""

@app.get("/")
def read_root():
    return RedirectResponse(url="/ui")

@app.get("/ui")
def serve_ui():
    return HTMLResponse(content=html_content)

@app.get("/api/status")
def get_status():
    load_dotenv()
    is_paper = os.getenv("PAPER_MODE", "true").lower() == "true"

    # Check websocket connection status
    ws_connected = False
    try:
        from core.status import get_ws_status
        ws_connected = get_ws_status()
    except Exception:
        pass

    return {
        "mode": "paper" if is_paper else "live",
        "ws_connected": ws_connected
    }

@app.get("/api/pnl")
def get_pnl():
    db = SessionLocal()
    try:
        summary = db.query(PnlSummary).filter(PnlSummary.date == date.today()).first()
        pnl = summary.total_pnl if summary else 0.0
        return {"pnl": pnl}
    finally:
        db.close()

@app.get("/api/trades")
def get_trades():
    db = SessionLocal()
    try:
        trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(50).all()
        return [{"symbol": t.symbol, "side": t.side, "quantity": t.quantity, "price": t.price, "timestamp": t.timestamp.isoformat()} for t in trades]
    finally:
        db.close()

@app.get("/api/orders")
def get_orders():
    db = SessionLocal()
    try:
        orders = db.query(Order).order_by(Order.timestamp.desc()).limit(50).all()
        return [{"symbol": o.symbol, "side": o.side, "quantity": o.quantity, "status": o.status, "timestamp": o.timestamp.isoformat()} for o in orders]
    finally:
        db.close()

@app.get("/api/positions")
def get_positions():
    db = SessionLocal()
    try:
        trades = db.query(Trade).all()
        positions = {}
        for t in trades:
            if t.symbol not in positions:
                positions[t.symbol] = 0
            if t.side == "BUY":
                positions[t.symbol] += t.quantity
            else:
                positions[t.symbol] -= t.quantity
        return {k: v for k, v in positions.items() if v != 0}
    finally:
        db.close()

@app.get("/api/market_data")
def get_market_data():
    db = SessionLocal()
    try:
        # Get the latest row for each symbol using a subquery grouping by symbol
        subquery = db.query(
            MarketData.symbol,
            func.max(MarketData.timestamp).label("max_timestamp")
        ).group_by(MarketData.symbol).subquery()

        latest_data = db.query(MarketData).join(
            subquery,
            (MarketData.symbol == subquery.c.symbol) &
            (MarketData.timestamp == subquery.c.max_timestamp)
        ).all()

        results = []
        for md in latest_data:
            results.append({
                "symbol": md.symbol,
                "trading_symbol": md.trading_symbol,
                "instrument_type": md.instrument_type,
                "strike_price": md.strike_price,
                "ltp": md.last_traded_price,
                "bid": md.bid_price,
                "ask": md.ask_price,
                "volume": md.volume,
                "oi": md.oi
            })
        return results
    finally:
        db.close()
