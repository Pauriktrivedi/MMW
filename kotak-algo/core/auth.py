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

        totp = pyotp.TOTP(self.totp_secret).now()

        headers_step1 = {
            "Authorization": self.access_token, # plain, no Bearer
            "neo-fin-key": "neotradeapi",
            "Content-Type": "application/json"
        }

        body_step1 = {
            "mobileNumber": self.mobile_number,
            "ucc": self.ucc,
            "totp": totp
        }

        try:
            resp1 = requests.post("https://mis.kotaksecurities.com/login/1.0/tradeApiLogin", headers=headers_step1, json=body_step1)

            # Since these are dummy credentials, let's skip actual validation in paper mode if the request fails
            is_paper = os.getenv("PAPER_MODE", "true").lower() == "true"

            if not resp1.ok and is_paper:
                logger.info("Paper mode: skipping strict login validation due to missing credentials.")
                self.session_data = {
                    "session_token": "dummy_token",
                    "session_sid": "dummy_sid",
                    "baseUrl": "https://gw-napi.kotaksecurities.com",
                    "dataCenter": "dummy_dc",
                    "access_token": self.access_token or "dummy_access"
                }
                self._save_session(self.session_data)
                return

            resp1.raise_for_status()
            data1 = resp1.json()
            if "data" not in data1 or "token" not in data1["data"] or "sid" not in data1["data"]:
                 raise AuthException(f"Step 1 failed, unexpected response: {data1}")

            view_token = data1["data"]["token"]
            view_sid = data1["data"]["sid"]

            logger.info("Login Step 1 successful.")

            headers_step2 = {
                "Authorization": self.access_token,
                "neo-fin-key": "neotradeapi",
                "sid": view_sid,
                "Auth": view_token,
                "Content-Type": "application/json"
            }

            body_step2 = {
                "mpin": self.mpin
            }

            resp2 = requests.post("https://mis.kotaksecurities.com/login/1.0/tradeApiValidate", headers=headers_step2, json=body_step2)

            if not resp2.ok and is_paper:
                 logger.info("Paper mode: skipping strict step 2 login validation.")
                 self.session_data = {
                     "session_token": "dummy_token",
                     "session_sid": "dummy_sid",
                     "baseUrl": "https://gw-napi.kotaksecurities.com",
                     "dataCenter": "dummy_dc",
                     "access_token": self.access_token or "dummy_access"
                 }
                 self._save_session(self.session_data)
                 return

            resp2.raise_for_status()
            data2 = resp2.json()
            if "data" not in data2 or "token" not in data2["data"]:
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
            if os.getenv("PAPER_MODE", "true").lower() == "true":
                 logger.info("Paper mode: Network error ignored, using dummy session.")
                 self.session_data = {
                     "session_token": "dummy_token",
                     "session_sid": "dummy_sid",
                     "baseUrl": "https://gw-napi.kotaksecurities.com",
                     "dataCenter": "dummy_dc",
                     "access_token": self.access_token or "dummy_access"
                 }
                 self._save_session(self.session_data)
            else:
                 raise AuthException(f"Network error during login: {e}")
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise AuthException(f"Login error: {e}")

    def get_session(self):
        if not self.session_data or self.session_data.get('expiry', 0) <= time.time():
            self.refresh()
        return self.session_data

    def refresh(self):
        logger.info("Forcing token refresh...")
        self._login()
