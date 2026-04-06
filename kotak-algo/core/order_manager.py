import requests
import json
import logging
from datetime import datetime
from database.database import SessionLocal
from database.models import Order

logger = logging.getLogger(__name__)

class OrderException(Exception):
    pass

class OrderManager:
    def __init__(self, auth_session):
        self.auth_session = auth_session
        self.base_url = auth_session.get("baseUrl")
        self.headers = {
            "Auth": auth_session.get("session_token"),
            "Sid": auth_session.get("session_sid"),
            "neo-fin-key": "neotradeapi",
            "Content-Type": "application/x-www-form-urlencoded"
        }

    def _post(self, endpoint, payload):
        url = f"{self.base_url}{endpoint}"
        jData = json.dumps(payload)
        data = {"jData": jData}

        try:
            resp = requests.post(url, headers=self.headers, data=data)
            resp.raise_for_status()
            result = resp.json()
            if result.get("stat") != "Ok":
                emsg = result.get("emsg", "Unknown error")
                raise OrderException(f"API Error: {emsg}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in _post: {e}")
            raise OrderException(f"Network error: {e}")

    def _get(self, endpoint):
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            result = resp.json()
            if result.get("stat") != "Ok":
                emsg = result.get("emsg", "Unknown error")
                raise OrderException(f"API Error: {emsg}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in _get: {e}")
            raise OrderException(f"Network error: {e}")

    def place_order(self, symbol, trading_symbol, qty, side, exchange_seg, order_type='MKT', price='0', product='NRML', trigger_price='0', instrument_type='EQ', strike_price=None, expiry_date=None):
        endpoint = "/quick/order/rule/ms/place"

        # map side to B or S
        tt = "B" if side.upper() == "BUY" else "S"
        # map order type
        pt = order_type
        if order_type == "MARKET":
            pt = "MKT"
        elif order_type == "LIMIT":
            pt = "L"
        elif order_type == "SL":
            pt = "SL"
        elif order_type == "SL-M":
            pt = "SL-M"

        payload = {
            "am": "NO",
            "dq": 0,
            "es": exchange_seg,
            "mp": 0,
            "pc": product,
            "pf": "N",
            "pr": str(price),
            "pt": pt,
            "qt": str(qty),
            "rt": "DAY",
            "tp": str(trigger_price),
            "ts": trading_symbol,
            "tt": tt
        }

        try:
            result = self._post(endpoint, payload)
            nOrdNo = result.get("nOrdNo", "")

            # Save to DB
            db = SessionLocal()
            try:
                new_order = Order(
                    symbol=symbol,
                    quantity=int(qty),
                    price=float(price) if price != '0' else None,
                    order_type=order_type,
                    side=side.upper(),
                    status="PENDING", # Kotak will push status updates via socket or we poll
                    instrument_type=instrument_type,
                    strike_price=strike_price,
                    expiry_date=expiry_date,
                    exchange_seg=exchange_seg,
                    trading_symbol=trading_symbol,
                    mode="live",
                    kotak_order_id=nOrdNo,
                    timestamp=datetime.now()
                )
                db.add(new_order)
                db.commit()
            finally:
                db.close()

            logger.info(f"Placed live order {nOrdNo} for {side} {qty} {trading_symbol}")
            return nOrdNo

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def modify_order(self, order_no, trading_symbol, qty, price, order_type, exchange_seg, product):
        endpoint = "/quick/order/vr/modify"

        pt = order_type
        if order_type == "MARKET":
            pt = "MKT"
        elif order_type == "LIMIT":
            pt = "L"

        payload = {
            "am": "NO",
            "on": order_no,
            "ts": trading_symbol,
            "qt": str(qty),
            "pr": str(price),
            "pt": pt,
            "es": exchange_seg,
            "pc": product,
            "rt": "DAY"
        }

        result = self._post(endpoint, payload)
        logger.info(f"Modified order {order_no}")
        return result

    def cancel_order(self, order_no):
        endpoint = "/quick/order/cancel"
        payload = {
            "on": order_no,
            "am": "NO"
        }
        result = self._post(endpoint, payload)
        logger.info(f"Cancelled order {order_no}")
        return result

    def get_order_book(self):
        return self._get("/quick/user/orders")

    def get_positions(self):
        return self._get("/quick/user/positions")

    def get_trade_book(self):
        return self._get("/quick/user/trades")

    def check_margin(self, token, exchange_seg, qty, price, order_type, product, side):
        endpoint = "/quick/user/check-margin"
        tt = "B" if side.upper() == "BUY" else "S"
        payload = {
            "brkName": "KOTAK",
            "brnchId": "ONLINE",
            "es": exchange_seg,
            "tk": str(token),
            "qt": str(qty),
            "pr": str(price),
            "pt": order_type,
            "pc": product,
            "tt": tt
        }
        return self._post(endpoint, payload)
