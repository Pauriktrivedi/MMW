from database.database import SessionLocal
from database.models import Order, Trade, MarketData, Signal, PnlSummary, RiskState
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_db():
    db = SessionLocal()
    try:
        # Just run a simple count to see if tables exist and are querying correctly
        order_count = db.query(Order).count()
        trade_count = db.query(Trade).count()
        logger.info(f"DB Check: Found {order_count} orders and {trade_count} trades in database.")

        # Test connection by executing raw SQL
        result = db.execute(text("SELECT 1")).fetchone()
        if result and result[0] == 1:
            logger.info("Database connection and basic queries successful.")
        else:
            logger.error("Database connection query failed.")
    except Exception as e:
        logger.error(f"Database check failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_db()
