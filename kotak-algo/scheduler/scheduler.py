from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import logging
from datetime import datetime
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class TradingScheduler:
    def __init__(self, system_controller):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Kolkata'))
        self.system_controller = system_controller

        # Hardcoded NSE holidays (example for 2024-2025)
        self.holidays = [
            "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
            "2024-04-11", "2024-04-17", "2024-05-01", "2024-06-17",
            "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-01",
            "2024-11-15", "2024-12-25",
            # Add 2025 holidays
            "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
            "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
            "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21",
            "2025-10-22", "2025-11-05", "2025-12-25"
        ]

    def is_holiday(self):
        today_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d")
        return today_str in self.holidays

    def start_system_job(self):
        if self.is_holiday():
            logger.info("Today is a holiday. Skipping system start.")
            return
        logger.info("Starting trading system for the day...")
        self.system_controller.start()

    def stop_system_job(self):
        if self.is_holiday():
            return
        logger.info("Stopping trading system for the day...")
        self.system_controller.stop()

    def generate_report_job(self):
        if self.is_holiday():
            return
        logger.info("Generating daily PnL report...")
        self.system_controller.generate_report()

    def start(self):
        # 9:00 AM Mon-Fri
        self.scheduler.add_job(self.start_system_job, 'cron', day_of_week='mon-fri', hour=9, minute=0)

        # 3:35 PM Mon-Fri
        self.scheduler.add_job(self.stop_system_job, 'cron', day_of_week='mon-fri', hour=15, minute=35)

        # 4:00 PM Mon-Fri
        self.scheduler.add_job(self.generate_report_job, 'cron', day_of_week='mon-fri', hour=16, minute=0)

        self.scheduler.start()
        logger.info("Scheduler started.")

    def stop(self):
        self.scheduler.shutdown()
        logger.info("Scheduler stopped.")
