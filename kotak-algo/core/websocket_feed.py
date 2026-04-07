import asyncio
import websockets
import json
import logging
import requests
import os
import threading
from datetime import datetime
from database.database import SessionLocal
from database.models import MarketData
import pandas as pd

logger = logging.getLogger(__name__)

class WebSocketFeedHandler:
    def __init__(self, auth_session, instrument_tokens, on_tick_callback=None, instruments=None):
        self.auth_session = auth_session
        self.instrument_tokens = instrument_tokens
        self.on_tick_callback = on_tick_callback
        self.instruments = instruments
        self.session_token = auth_session.get("session_token")
        self.session_sid = auth_session.get("session_sid")
        self.data_center = auth_session.get("dataCenter")
        self.ws_url = os.getenv("KOTAK_WS_URL", "wss://mlhsm.kotaksecurities.com")
        self.running = False
        self._loop = None
        self._thread = None

    def start(self):
        self.running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_in_thread, daemon=True)
        self._thread.start()
        logger.info("WebSocket Feed Handler started.")

    def _run_in_thread(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_loop())

    def stop(self):
        self.running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("WebSocket Feed Handler stopped.")

    async def _run_loop(self):
        retry_delays = [2, 4, 8, 16, 30, 60]
        retry_idx = 0

        while self.running:
            try:
                await self._connect()
                retry_idx = 0 # reset on successful disconnect that wasn't an exception
            except asyncio.CancelledError:
                logger.info("Websocket task cancelled.")
                break
            except Exception as e:
                logger.error(f"Websocket connection error: {e}")
                delay = retry_delays[retry_idx]
                logger.info(f"Reconnecting in {delay} seconds...")
                await asyncio.sleep(delay)
                if retry_idx < len(retry_delays) - 1:
                    retry_idx += 1

    async def _connect(self):
        # The Neo websocket usually expects query params or initial auth payload.
        # This is a generic implementation matching the standard neo flow:
        ws_url_full = f"{self.ws_url}?sid={self.session_sid}&token={self.session_token}&dc={self.data_center}"

        async with websockets.connect(ws_url_full) as ws:
            logger.info("WebSocket connected successfully.")

            # Subscribe
            subs_str = "&".join(self.instrument_tokens) + "&"
            sub_payload = {
                "type": "sub",
                "scrips": subs_str,
                "channel": "1" # usually 1 for market data
            }
            await ws.send(json.dumps(sub_payload))
            logger.info(f"Sent subscription payload for {len(self.instrument_tokens)} instruments.")

            while self.running:
                message = await ws.recv()
                try:
                    data = json.loads(message)
                    self._store_tick(data)
                except json.JSONDecodeError:
                    # Sometimes heartbeat or raw binary is sent, depending on specific endpoint
                    pass
                except Exception as e:
                    logger.error(f"Error processing tick: {e}")

    def _store_tick(self, data):
        # Typical Kotak response structure needs parsing. Assuming a standard structure here:
        if isinstance(data, list):
            for tick in data:
                self._process_single_tick(tick)
        else:
            self._process_single_tick(data)

    def _process_single_tick(self, tick):
        # Forward tick to registered strategy
        if self.on_tick_callback:
            try:
                self.on_tick_callback(tick)
            except Exception as e:
                logger.error(f"Error executing on_tick callback: {e}")

        # Ex: {"tk": "11536", "e": "nse_cm", "ltp": "150.5", "v": "1000", "bp1": "150.4", "sp1": "150.6", "oi": "0"}
        if "tk" not in tick:
             return

        token = str(tick.get("tk"))
        exchange_seg = tick.get("e", "")
        symbol = f"{exchange_seg}|{token}"

        # Determine actual details using InstrumentMaster
        trading_symbol = symbol
        instrument_type = "EQ"
        strike_price = None
        expiry_date = None

        if self.instruments:
            # Check FO
            if exchange_seg == 'nse_fo' and self.instruments.fo_df is not None:
                match = self.instruments.fo_df[self.instruments.fo_df['pSymbol'].astype(str) == token]
                if not match.empty:
                    row = match.iloc[0]
                    trading_symbol = str(row['pTrdSymbol'])
                    instrument_type = str(row['pOptionType']) if 'OPT' in str(row['pInstrumentType']) else "FUT"
                    strike_price = float(row['dStrikePrice']) if not pd.isna(row['dStrikePrice']) else None
                    expiry_date = str(row['lExpiryDate'])
            # Check CM
            elif exchange_seg == 'nse_cm' and self.instruments.cm_df is not None:
                match = self.instruments.cm_df[self.instruments.cm_df['pSymbol'].astype(str) == token]
                if not match.empty:
                    trading_symbol = str(match.iloc[0]['pTrdSymbol'])

        db = SessionLocal()
        try:
            md = MarketData(
                symbol=symbol,
                trading_symbol=trading_symbol,
                exchange_seg=exchange_seg,
                instrument_type=instrument_type,
                bid_price=float(tick.get("bp1", tick.get("ltp", 0))),
                ask_price=float(tick.get("sp1", tick.get("ltp", 0))),
                last_traded_price=float(tick.get("ltp", 0)),
                volume=int(tick.get("v", 0)),
                oi=int(tick.get("oi", 0)),
                strike_price=strike_price,
                expiry_date=expiry_date,
                timestamp=datetime.now()
            )
            db.add(md)
            db.commit()
        except Exception as e:
            logger.error(f"DB Error storing tick: {e}")
        finally:
            db.close()


class QuotesClient:
    def __init__(self, auth_session):
        self.access_token = auth_session.get("access_token")
        self.base_url = auth_session.get("baseUrl")
        self.headers = {
            "Authorization": self.access_token
        }

    def get_quote(self, exchange_seg, symbol):
        url = f"{self.base_url}/script-details/1.0/quotes/neosymbol/{exchange_seg}|{symbol}/all"
        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching quote: {e}")
            return None
