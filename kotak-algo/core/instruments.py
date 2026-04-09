import requests
import pandas as pd
import os
import time
import logging

logger = logging.getLogger(__name__)

class InstrumentMaster:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.cm_file = os.path.join(self.data_dir, "nse_cm.csv")
        self.fo_file = os.path.join(self.data_dir, "nse_fo.csv")
        self.cm_df = None
        self.fo_df = None

    def _is_stale(self, filepath):
        if not os.path.exists(filepath):
            return True
        file_age = time.time() - os.path.getmtime(filepath)
        return file_age > 86400  # older than 1 day

    def download(self, auth_session):
        logger.info("Checking instrument files...")

        headers = {
            "Authorization": auth_session["access_token"]
        }

        baseUrl = auth_session["baseUrl"]
        url = f"{baseUrl}/script-details/1.0/masterscrip/file-paths"

        try:
            # Only download if stale
            if self._is_stale(self.cm_file) or self._is_stale(self.fo_file):
                logger.info("Downloading master scrip files...")
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                file_paths = data.get("data", {}).get("filesPaths", [])
                for file_url in file_paths:
                    if "nse_cm" in file_url:
                        csv_data = requests.get(file_url).content
                        with open(self.cm_file, 'wb') as f:
                            f.write(csv_data)
                        logger.info("Downloaded nse_cm.csv")
                    elif "nse_fo" in file_url:
                        csv_data = requests.get(file_url).content
                        with open(self.fo_file, 'wb') as f:
                            f.write(csv_data)
                        logger.info("Downloaded nse_fo.csv")

                logger.info("Download complete.")
            else:
                logger.info("Instrument files are up to date.")
        except Exception as e:
            logger.error(f"Error downloading instruments: {e}")

    def load(self, segment='nse_fo'):
        file_path = self.fo_file if segment == 'nse_fo' else self.cm_file
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found. Please download first.")
            return None

        try:
            df = pd.read_csv(file_path)
            if segment == 'nse_fo':
                self.fo_df = df
            else:
                self.cm_df = df
            return df
        except Exception as e:
            logger.error(f"Error loading {segment} csv: {e}")
            return None

    def find_option(self, underlying, expiry, strike, opt_type):
        if self.fo_df is None:
            self.load('nse_fo')

        if self.fo_df is None:
            return None

        try:
            df = self.fo_df
            # Match underlying (pSymbolName), expiry (lExpiryDate), strike (dStrikePrice), opt_type (pOptionType - CE/PE)
            # Column names depend on exact CSV structure, using expected names based on neo API
            mask = (
                (df['pSymbolName'].str.contains(underlying, case=False, na=False)) &
                (df['lExpiryDate '].astype(str) == str(expiry)) &
                (df['dStrikePrice;'].astype(float) == float(strike)) &
                (df['pOptionType'].str.upper() == opt_type.upper()) &
                (df['pInstType'].str.contains("OPT", na=False))
            )
            result = df[mask]

            if not result.empty:
                row = result.iloc[0]
                return {
                    "pTrdSymbol": row['pTrdSymbol'],
                    "pSymbol": str(row['pSymbol'])
                }
            return None
        except Exception as e:
            logger.error(f"Error finding option: {e}")
            return None

    def find_future(self, underlying, expiry):
        if self.fo_df is None:
            self.load('nse_fo')

        if self.fo_df is None:
            return None

        try:
            df = self.fo_df
            mask = (
                (df['pSymbolName'].str.contains(underlying, case=False, na=False)) &
                (df['lExpiryDate '].astype(str) == str(expiry)) &
                (df['pInstType'].str.contains("FUT", na=False))
            )
            result = df[mask]

            if not result.empty:
                row = result.iloc[0]
                return {
                    "pTrdSymbol": row['pTrdSymbol'],
                    "pSymbol": str(row['pSymbol'])
                }
            return None
        except Exception as e:
            logger.error(f"Error finding future: {e}")
            return None
