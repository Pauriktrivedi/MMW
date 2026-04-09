import logging
from datetime import datetime, time
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TwelveThirtyFiveStrategy(BaseStrategy):
    """
    Executes a short ATM Call and 50-point ITM Put exactly at 12:35 PM daily.
    Applies a strict 25-point independent Stop Loss (SL) on both legs based on the 12:35 LTP.
    Exits any remaining positions at 03:25 PM.
    """
    def __init__(self, mode='paper', instrument_master=None, underlying_symbol="NIFTY", qty=50):
        super().__init__(name="12:35_Options_Selling", mode=mode)
        self.instrument_master = instrument_master
        self.underlying_symbol = underlying_symbol
        self.qty = qty

        self.executed_today = False
        self.positions = {} # symbol: {'type': 'CE'/'PE', 'entry_price': X, 'sl_price': Y, 'status': 'OPEN'/'CLOSED'}

        self.entry_time = time(12, 35)
        self.exit_time = time(15, 25)
        self.sl_points = 25.0

        # Track LTP to use for execution at 12:35
        self.current_underlying_price = 0.0

    def get_time_from_tick(self, tick_data):
        timestamp = tick_data.get('timestamp')
        if isinstance(timestamp, str):
             # Try parsing
             try:
                 dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                 return dt.time()
             except Exception:
                 return datetime.now().time()
        elif isinstance(timestamp, datetime):
            return timestamp.time()
        return datetime.now().time()

    def _get_strikes(self, ltp):
        # Round to nearest 50 for Nifty
        atm_strike = round(ltp / 50) * 50
        # 50-point ITM PE (strike higher than ATM)
        itm_pe_strike = atm_strike + 50
        return atm_strike, itm_pe_strike

    def _execute_entry(self, tick_data):
        if not self.instrument_master:
            logger.error("InstrumentMaster not provided to 12:35 strategy.")
            return

        ltp = self.current_underlying_price
        if ltp == 0:
            logger.warning("Underlying price is 0, cannot execute 12:35 entry.")
            return

        atm_strike, itm_pe_strike = self._get_strikes(ltp)
        logger.info(f"[12:35 Entry] Underlying LTP: {ltp}, ATM CE Strike: {atm_strike}, ITM PE Strike: {itm_pe_strike}")

        # Get expiry (just use current date for simulation or require logic to find next weekly expiry)
        # Assuming we can find the nearest options via InstrumentMaster logic
        # For simplicity, we just look up the strikes without specifying an exact expiry date to get the nearest
        # A robust system would specify the exact weekly expiry

        # Example logic to find the tokens (requires instrument master to be loaded)
        ce_token = f"OPT_{self.underlying_symbol}_{atm_strike}_CE" # Mock token if find_option fails
        pe_token = f"OPT_{self.underlying_symbol}_{itm_pe_strike}_PE"

        # In a real environment, we use instrument_master.find_option
        if self.instrument_master and self.instrument_master.fo_df is not None:
             try:
                 df = self.instrument_master.fo_df
                 symbol_str = self.underlying_symbol.split('|')[-1] if '|' in self.underlying_symbol else self.underlying_symbol
                 if symbol_str == "Nifty 50":
                     symbol_str = "NIFTY"

                 mask = (
                     (df['pSymbolName'].str.upper() == symbol_str.upper()) &
                     (df['pInstType'].str.contains('OPT', na=False))
                 )
                 opts = df[mask]
                 if not opts.empty:
                     expiries = opts['lExpiryDate'].dropna().unique()
                     if len(expiries) > 0:
                         nearest_expiry = sorted(expiries)[0]
                         current_opts = opts[opts['lExpiryDate'] == nearest_expiry]

                         ce_opts = current_opts[(current_opts['dStrikePrice'] == atm_strike) & (current_opts['pOptionType'] == 'CE')]
                         if not ce_opts.empty:
                             ce_token = f"nse_fo|{ce_opts.iloc[0]['pSymbol']}"

                         pe_opts = current_opts[(current_opts['dStrikePrice'] == itm_pe_strike) & (current_opts['pOptionType'] == 'PE')]
                         if not pe_opts.empty:
                             pe_token = f"nse_fo|{pe_opts.iloc[0]['pSymbol']}"
             except Exception as e:
                 logger.error(f"Error finding tokens dynamically: {e}")

        # Since this is selling options, side is SELL
        # 1. Sell ATM CE
        ce_order_id = self.place_order(
            symbol=ce_token,
            trading_symbol=ce_token,
            qty=self.qty,
            side='SELL',
            exchange_seg='nse_fo',
            order_type='MARKET',
            instrument_type='CE'
        )

        # 2. Sell 50-pt ITM PE
        pe_order_id = self.place_order(
            symbol=pe_token,
            trading_symbol=pe_token,
            qty=self.qty,
            side='SELL',
            exchange_seg='nse_fo',
            order_type='MARKET',
            instrument_type='PE'
        )

        self.executed_today = True

    def _execute_exit_all(self):
        logger.info("[15:25 Auto-Exit] Exiting all remaining positions.")
        for symbol, pos in self.positions.items():
            if pos['status'] == 'OPEN':
                self.place_order(
                    symbol=symbol,
                    trading_symbol=symbol,
                    qty=self.qty,
                    side='BUY', # Cover short
                    exchange_seg='nse_fo',
                    order_type='MARKET'
                )
                pos['status'] = 'CLOSED'

    def on_tick(self, tick_data: dict):
        symbol = tick_data.get('symbol', '')
        ltp = tick_data.get('last_traded_price', 0.0)
        tick_time = self.get_time_from_tick(tick_data)

        # Track underlying price
        if symbol == self.underlying_symbol or "NIFTY 50" in symbol.upper():
             self.current_underlying_price = ltp

        # 12:35 PM Entry Check
        if not self.executed_today and tick_time >= self.entry_time and tick_time < time(12, 40):
            self._execute_entry(tick_data)

        # Stop Loss Management for Open Positions
        for pos_symbol, pos in self.positions.items():
            if pos['status'] == 'OPEN' and pos_symbol == symbol:
                # Since we sold options, loss occurs when LTP > entry_price
                if ltp >= pos['sl_price']:
                    logger.info(f"[Stop Loss Hit] {pos_symbol} LTP {ltp} >= SL {pos['sl_price']}. Covering short.")
                    self.place_order(
                        symbol=pos_symbol,
                        trading_symbol=pos_symbol,
                        qty=self.qty,
                        side='BUY', # Cover short
                        exchange_seg='nse_fo',
                        order_type='MARKET'
                    )
                    pos['status'] = 'CLOSED'

        # 15:25 PM Exit Check
        if self.executed_today and tick_time >= self.exit_time:
            # Ensure we only exit once by checking if any positions are still open
            if any(pos['status'] == 'OPEN' for pos in self.positions.values()):
                self._execute_exit_all()

    def on_signal(self, signal_data: dict):
        pass

    def on_order_fill(self, trade_data: dict):
        symbol = trade_data['symbol']
        side = trade_data['side']
        fill_price = trade_data['price']

        if side == 'SELL':
            # This is an entry
            self.positions[symbol] = {
                'entry_price': fill_price,
                'sl_price': fill_price + self.sl_points,
                'status': 'OPEN'
            }
            logger.info(f"[Position Opened] {symbol} sold at {fill_price}, SL set at {fill_price + self.sl_points}")
        elif side == 'BUY':
            # This is an exit
            if symbol in self.positions:
                self.positions[symbol]['status'] = 'CLOSED'
                logger.info(f"[Position Closed] {symbol} covered at {fill_price}")
