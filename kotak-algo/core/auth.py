import requests
import pyotp
import os
import time
import json
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AuthException(Exception):
    pass

class KotakNeoAuth:
    def __init__(self):
        load_dotenv()
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.mobile_number = os.getenv("MOBILE_NUMBER")
        self.ucc = os.getenv("UCC")
        self.mpin = os.getenv("MPIN")
        self.totp_secret = os.getenv("TOTP_SECRET")
        self.session_file = "session.json"

        if not all([self.access_token, self.mobile_number, self.ucc, self.mpin, self.totp_secret]):
            logger.warning("Not all Kotak credentials found in .env. Auth will fail if live.")

        self.session_data = self._load_session()
        if not self.session_data:
            self._login()

    def _load_session(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    if data.get('expiry', 0) > time.time():
                        logger.info("Loaded valid session from cache.")
                        return data
                    else:
                        logger.info("Cached session expired.")
            except Exception as e:
                logger.error(f"Error loading session file: {e}")
        return None

    def _save_session(self, data):
        data['expiry'] = time.time() + 3600  # expire in 1 hour
        try:
            with open(self.session_file, 'w') as f:
                json.dump(data, f)
            logger.info("Session saved to cache.")
        except Exception as e:
            logger.error(f"Error saving session file: {e}")

    def _login(self):
        logger.info("Starting Kotak Neo 2-step login process...")

        if not self.totp_secret:
            raise AuthException("TOTP secret is missing. Cannot proceed with login.")

        try:
            totp = pyotp.TOTP(self.totp_secret).now()
        except Exception as e:
            raise AuthException(f"Invalid TOTP_SECRET format: {e}")

        headers_step1 = {
            "Authorization": f"Bearer {self.access_token}",
            "neo-fin-key": "neotradeapi",
            "Content-Type": "application/json"
        }

        body_step1 = {
            "mobileNumber": self.mobile_number,
            "ucc": self.ucc,
            "totp": totp
        }

        try:
            logger.info(f"Step 1 Headers: {headers_step1}")
            logger.info(f"Step 1 Payload: {body_step1}")
            resp1 = requests.post("https://mis.kotaksecurities.com/login/1.0/tradeApiLogin", headers=headers_step1, json=body_step1)
            resp1.raise_for_status()
            data1 = resp1.json()
            if not data1.get("data") or "token" not in data1.get("data", {}) or "sid" not in data1.get("data", {}):
                 raise AuthException(f"Step 1 failed, unexpected response: {data1}")

            view_token = data1["data"]["token"]
            view_sid = data1["data"]["sid"]

            logger.info("Login Step 1 successful.")

            headers_step2 = {
                "Authorization": f"Bearer {self.access_token}",
                "neo-fin-key": "neotradeapi",
                "sid": view_sid,
                "Auth": view_token,
                "Content-Type": "application/json"
            }

            body_step2 = {
                "mpin": self.mpin
            }

            logger.info(f"Step 2 Headers: {headers_step2}")
            logger.info(f"Step 2 Payload: {body_step2}")
            resp2 = requests.post("https://mis.kotaksecurities.com/login/1.0/tradeApiValidate", headers=headers_step2, json=body_step2)
            resp2.raise_for_status()
            data2 = resp2.json()
            if not data2.get("data") or "token" not in data2.get("data", {}):
                 raise AuthException(f"Step 2 failed, unexpected response: {data2}")

            self.session_data = {
                "session_token": data2["data"]["token"],
                "session_sid": data2["data"]["sid"],
                "baseUrl": data2["data"].get("baseUrl", "https://gw-napi.kotaksecurities.com"),
                "dataCenter": data2["data"].get("dataCenter", ""),
                "access_token": self.access_token
            }

            self._save_session(self.session_data)
            logger.info("Login Step 2 successful. Authenticated.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during login: {e}")
            raise AuthException(f"Network error during login: {e}")
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise AuthException(f"Login error: {e}")

    def get_session(self):
        if not self.session_data or self.session_data.get('expiry', 0) <= time.time():
            self.refresh()
        return self.session_data

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.access_token}",
            "neo-fin-key": "neotradeapi",
            "sid": self.session_data.get("session_sid") if self.session_data else "",
            "Auth": self.session_data.get("session_token") if self.session_data else ""
        }

    def refresh(self):
        logger.info("Forcing token refresh...")
        self._login()
