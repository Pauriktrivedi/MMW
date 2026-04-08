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
    <title>Paurik Trivedi's Algo Dashboard</title>
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
        .strategy-container { display: flex; flex-direction: column; gap: 15px; }
        .strategy-summary { display: flex; gap: 20px; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; flex: 1; text-align: center; border: 1px solid #ddd; }
        .stat-box .num { font-size: 1.5em; font-weight: bold; }
        .stat-box.live-box .num { color: #4caf50; }
        .stat-box.paused-box .num { color: #ff9800; }
        .stat-box.stopped-box .num { color: #f44336; }
        .strategy-details { background: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; display: none; }
        .strategy-details p { margin: 5px 0; }
        select { padding: 10px; font-size: 1em; border-radius: 5px; border: 1px solid #ccc; width: 100%; max-width: 400px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Paurik Trivedi's Algo Dashboard</h1>
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
        <h2>Strategy Management</h2>
        <div class="strategy-container">
            <div class="strategy-summary">
                <div class="stat-box live-box">
                    <div>Live</div>
                    <div id="stat-live" class="num">0</div>
                </div>
                <div class="stat-box paused-box">
                    <div>Paused</div>
                    <div id="stat-paused" class="num">0</div>
                </div>
                <div class="stat-box stopped-box">
                    <div>Stopped</div>
                    <div id="stat-stopped" class="num">0</div>
                </div>
            </div>

            <div>
                <label for="strategy-select"><strong>Select Strategy:</strong></label><br/>
                <select id="strategy-select">
                    <option value="">-- Choose a Strategy --</option>
                </select>
            </div>

            <div id="strategy-details-panel" class="strategy-details">
                <h3 id="det-name" style="margin-top: 0;">Strategy Name</h3>
                <p><strong>Status:</strong> <span id="det-status" class="badge" style="color: black;">Unknown</span></p>
                <p><strong>Description:</strong> <span id="det-desc"></span></p>
                <p><strong>Schedule:</strong> <span id="det-sched"></span></p>
                <p><strong>Details:</strong> <span id="det-details"></span></p>
            </div>
        </div>
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
        let strategiesData = [];

        async function fetchStrategies() {
            try {
                const res = await fetch('/api/strategies');
                strategiesData = await res.json();

                let liveCount = 0;
                let pausedCount = 0;
                let stoppedCount = 0;

                const selectEl = document.getElementById('strategy-select');
                const currentValue = selectEl.value;

                // We only want to populate the dropdown once or if the length changes
                // to avoid resetting the user's selection on every poll
                if (selectEl.options.length <= 1 || selectEl.options.length - 1 !== strategiesData.length) {
                    selectEl.innerHTML = '<option value="">-- Choose a Strategy --</option>';
                    strategiesData.forEach(s => {
                        const opt = document.createElement('option');
                        opt.value = s.id;
                        opt.textContent = s.name + " (" + s.status + ")";
                        selectEl.appendChild(opt);
                    });
                    if (currentValue) selectEl.value = currentValue;
                }

                strategiesData.forEach(s => {
                    if (s.status.toLowerCase() === 'live') liveCount++;
                    else if (s.status.toLowerCase() === 'paused') pausedCount++;
                    else stoppedCount++;

                    // Update text in dropdown if status changed
                    const optToUpdate = Array.from(selectEl.options).find(o => o.value === s.id);
                    if (optToUpdate) {
                         optToUpdate.textContent = s.name + " (" + s.status + ")";
                    }
                });

                document.getElementById('stat-live').textContent = liveCount;
                document.getElementById('stat-paused').textContent = pausedCount;
                document.getElementById('stat-stopped').textContent = stoppedCount;

                // If one is selected, update its details
                updateStrategyDetails();

            } catch (e) {
                console.error("Error fetching strategies", e);
            }
        }

        function updateStrategyDetails() {
            const selectEl = document.getElementById('strategy-select');
            const panel = document.getElementById('strategy-details-panel');
            const selectedId = selectEl.value;

            if (!selectedId) {
                panel.style.display = 'none';
                return;
            }

            const strategy = strategiesData.find(s => s.id === selectedId);
            if (strategy) {
                panel.style.display = 'block';
                document.getElementById('det-name').textContent = strategy.name;
                document.getElementById('det-desc').textContent = strategy.description;
                document.getElementById('det-sched').textContent = strategy.schedule;
                document.getElementById('det-details').textContent = strategy.details;

                const statusEl = document.getElementById('det-status');
                statusEl.textContent = strategy.status;
                if (strategy.status.toLowerCase() === 'live') {
                    statusEl.style.backgroundColor = '#4caf50';
                    statusEl.style.color = 'white';
                } else if (strategy.status.toLowerCase() === 'paused') {
                    statusEl.style.backgroundColor = '#ff9800';
                    statusEl.style.color = 'white';
                } else {
                    statusEl.style.backgroundColor = '#f44336';
                    statusEl.style.color = 'white';
                }
            }
        }

        document.getElementById('strategy-select').addEventListener('change', updateStrategyDetails);

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

        setInterval(() => {
            fetchData();
            fetchStrategies();
        }, 5000);

        fetchData();
        fetchStrategies();
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

@app.get("/api/strategies")
def get_strategies():
    # In a fully dynamic system, this would be read from a database or registry.
    # For now, we return the known configured strategies.
    import os
    from dotenv import load_dotenv
    load_dotenv()

    active_strategy = os.getenv("ACTIVE_STRATEGY", "twelve_thirty_five")

    strategies = [
        {
            "id": "breakout_range",
            "name": "Breakout Range Strategy",
            "description": "Executes trades when the underlying index breaks out of a predefined range.",
            "schedule": "Continuous during market hours.",
            "details": "Monitors NIFTY 50. Goes LONG if price breaks above Range High. Goes SHORT if price breaks below Range Low.",
            "status": "Live" if active_strategy == "breakout_range" else "Stopped"
        },
        {
            "id": "twelve_thirty_five",
            "name": "12:35 Options Selling",
            "description": "Shorts ATM CE and 50-pt ITM PE exactly at 12:35 PM daily.",
            "schedule": "Executes at 12:35 PM. Auto-exits at 03:25 PM.",
            "details": "Applies a strict 25-point independent Stop Loss (SL) on both legs based on the entry price. Recovers independently.",
            "status": "Live" if active_strategy == "twelve_thirty_five" else "Stopped"
        },
        {
            "id": "sample_strategy",
            "name": "Sample Dummy Strategy",
            "description": "A basic placeholder strategy that randomly prints ticks.",
            "schedule": "Continuous.",
            "details": "Used for testing the framework and websocket connections.",
            "status": "Live" if active_strategy == "sample_strategy" else "Stopped"
        }
    ]

    return strategies


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
