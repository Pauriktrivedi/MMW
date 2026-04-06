# Kotak Neo Algorithmic Trading System

A complete professional algorithmic trading system for Kotak Neo broker in Python. Runs locally, trades automatically during market hours, supports paper trading mode, and stores all data in a local SQLite database.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd kotak-algo
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```
   **Important:** Never share your `.env` file or commit it to version control. It contains your live credentials.

5. **TOTP Registration Steps:**
   - Go to your authenticator app (e.g., Google Authenticator, Authy).
   - Get the base32 secret code that you used to register Kotak Neo.
   - Put this code in `.env` under `TOTP_SECRET`.

## Running the Application

To run the application, start the main entry point:
```bash
python main.py
```
This will initialize the database, apply settings, start the background dashboard, and set up the scheduler to run your trading system during market hours.

## Running Tests

To verify that the system is fully operational, run the integration tests:
```bash
python -m pytest tests/
```

## Project Structure

- `core/`: Core Kotak Neo API integration (Auth, Websockets, Orders, Instruments).
- `database/`: Local SQLite database configuration and SQLAlchemy models.
- `paper_trade/`: Paper trading simulator engine.
- `risk/`: Risk management module (position sizing, loss limits).
- `strategies/`: Core strategy logic. Base classes and implementations.
- `analytics/`: Generating PnL summaries and metrics.
- `dashboard/`: FastAPI based simple web dashboard for monitoring.
- `scheduler/`: APScheduler integration to start/stop the system automatically.
- `data/`: Temporary data folder (instruments CSV).

## Switching between Paper and Live Mode

- `PAPER_MODE=true` in `.env`: All orders are simulated. None hit the live exchange.
- `PAPER_MODE=false` in `.env`: Orders are sent to the Kotak Neo API and hit the live exchange. **USE WITH CAUTION.**
