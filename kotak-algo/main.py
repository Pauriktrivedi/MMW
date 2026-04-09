import os
import sys
import time
import uvicorn
import logging
import threading
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

import pandas as pd
from database.database import init_db
from core.auth import KotakNeoAuth
from core.instruments import InstrumentMaster
from core.websocket_feed import WebSocketFeedHandler
from core.order_manager import OrderManager
from paper_trade.paper_trade import PaperTradeSimulator
from risk.risk_manager import RiskManager
from scheduler.scheduler import TradingScheduler
from analytics.pnl_report import PnlReport
from dashboard.dashboard import app as fastapi_app
from strategies.sample_strategy import SampleStrategy
from strategies.breakout_strategy import BreakoutRangeStrategy
from strategies.twelve_thirty_five import TwelveThirtyFiveStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("System")
console = Console()

class SystemController:
    def __init__(self):
        load_dotenv()
        self.mode = "paper" if os.getenv("PAPER_MODE", "true").lower() == "true" else "live"
        self.virtual_cash = os.getenv("VIRTUAL_CASH", "500000")
        self.max_daily_loss = os.getenv("MAX_DAILY_LOSS", "10000")
        self.max_positions = os.getenv("MAX_OPEN_POSITIONS", "5")

        self.auth = None
        self.instruments = None
        self.ws_handler = None
        self.order_manager = None
        self.paper_trader = None
        self.risk_manager = None
        self.strategy = None

    def start(self):
        logger.info(f"Initializing system in {self.mode.upper()} mode...")

        try:
            # 1. Init Risk
            self.risk_manager = RiskManager(
                max_daily_loss=self.max_daily_loss,
                per_trade_stop_loss=2000, # Example hardcoded
                max_open_positions=self.max_positions
            )

            # 2. Auth & Instruments
            self.auth = KotakNeoAuth()
            session = self.auth.get_session()

            self.instruments = InstrumentMaster()
            self.instruments.download(session)
            self.instruments.load('nse_fo')
            self.instruments.load('nse_cm')

            # 3. Setup Trading mode
            if self.mode == 'paper':
                self.paper_trader = PaperTradeSimulator()
                self.paper_trader.start()
            else:
                self.order_manager = OrderManager(session)

            # 4. Initialize Strategy
            strategy_name = os.getenv("ACTIVE_STRATEGY", "twelve_thirty_five")
            if strategy_name == "breakout_range":
                self.strategy = BreakoutRangeStrategy(
                    mode=self.mode,
                    paper_trader=self.paper_trader,
                    live_trader=self.order_manager,
                    risk_manager=self.risk_manager,
                    symbol="nse_cm|Nifty 50",
                    range_high=22010.0, # Example values for testing
                    range_low=21990.0,
                    quantity=50
                )
            elif strategy_name == "twelve_thirty_five":
                self.strategy = TwelveThirtyFiveStrategy(
                    mode=self.mode,
                    instrument_master=self.instruments,
                    underlying_symbol="nse_cm|Nifty 50"
                )
                self.strategy.paper_trader = self.paper_trader
                self.strategy.live_trader = self.order_manager
                self.strategy.risk_manager = self.risk_manager
            else:
                self.strategy = SampleStrategy(
                    mode=self.mode,
                    paper_trader=self.paper_trader,
                    live_trader=self.order_manager,
                    risk_manager=self.risk_manager
                )

            # 5. Setup Websocket
            # Subscribe to major indices using their names as required by Kotak Neo
            tokens = [
                "nse_cm|Nifty 50", "nse_cm|NIFTY BANK", "bse_cm|SENSEX"
            ]

            # Dynamically fetch correct option chain tokens using InstrumentMaster
            if self.instruments.fo_df is not None:
                try:
                    df = self.instruments.fo_df
                    # Filter for NIFTY options
                    mask = (
                        (df['pSymbolName'].str.upper() == 'NIFTY') &
                        (df['pInstType'].str.contains('OPT', na=False))
                    )
                    nifty_options = df[mask]
                    if not nifty_options.empty:
                        # Find the nearest expiry
                        expiries = nifty_options['lExpiryDate'].dropna().unique()
                        if len(expiries) > 0:
                            nearest_expiry = sorted(expiries)[0]
                            current_expiry_opts = nifty_options[nifty_options['lExpiryDate'] == nearest_expiry]

                            # Pick middle strikes
                            strikes = sorted(current_expiry_opts['dStrikePrice'].dropna().unique())
                            if strikes:
                                mid_idx = len(strikes) // 2
                                # Select 2 strikes around the middle
                                selected_strikes = strikes[max(0, mid_idx-1):mid_idx+1]

                                for strike in selected_strikes:
                                    # CE
                                    ce_opt = current_expiry_opts[(current_expiry_opts['dStrikePrice'] == strike) & (current_expiry_opts['pOptionType'] == 'CE')]
                                    if not ce_opt.empty:
                                        tokens.append(f"nse_fo|{ce_opt.iloc[0]['pSymbol']}")

                                    # PE
                                    pe_opt = current_expiry_opts[(current_expiry_opts['dStrikePrice'] == strike) & (current_expiry_opts['pOptionType'] == 'PE')]
                                    if not pe_opt.empty:
                                        tokens.append(f"nse_fo|{pe_opt.iloc[0]['pSymbol']}")

                    logger.info(f"Dynamically added live option tokens: {tokens[3:]}")
                except Exception as e:
                    logger.error(f"Error dynamically fetching option tokens: {e}")
            else:
                logger.warning("InstrumentMaster data not available. Using fallback dummy option tokens.")
                dummy_option_tokens = ["nse_fo|12345", "nse_fo|12346", "nse_fo|12347", "nse_fo|12348"]
                tokens.extend(dummy_option_tokens)

            self.ws_handler = WebSocketFeedHandler(session, tokens, on_tick_callback=self.strategy.on_tick, instruments=self.instruments)
            self.ws_handler.start()

            logger.info("System successfully started.")

        except Exception as e:
            logger.error(f"Failed to start system: {e}")

    def stop(self):
        logger.info("Stopping system...")
        if self.ws_handler:
            self.ws_handler.stop()
        if self.paper_trader:
            self.paper_trader.stop()

        # Calculate end of day PnL
        self.generate_report()
        logger.info("System stopped.")

    def generate_report(self):
        report = PnlReport()
        summary = report.calculate_daily_pnl()
        if summary:
            console.print(Panel(
                f"Total PnL: {summary.total_pnl}\nTrades: {summary.total_trades}",
                title="Daily PnL Report",
                expand=False,
                style="green" if summary.total_pnl >= 0 else "red"
            ))

def run_dashboard():
    # Run FastAPI app with uvicorn
    # Redirecting uvicorn access logs to standard logging can be tricky, using default here.
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")

def main():
    # Print Banner
    console.print(Panel.fit("[bold blue]Kotak Neo Algorithmic Trading System[/bold blue]", subtitle="Initializing..."))

    # Init DB
    init_db()

    # Init Controller
    controller = SystemController()

    console.print(f"Mode: [bold {'yellow' if controller.mode == 'paper' else 'red'}]{controller.mode.upper()}[/]")
    if controller.mode == 'paper':
         console.print(f"Virtual Cash: {controller.virtual_cash}")
    console.print(f"Risk Limits: Daily Loss = {controller.max_daily_loss}, Max Pos = {controller.max_positions}")

    # Start Dashboard
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    logger.info("Dashboard started on port 8000")

    # Start Scheduler
    scheduler = TradingScheduler(controller)
    scheduler.start()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        scheduler.stop()
        controller.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
