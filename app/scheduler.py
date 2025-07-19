from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from db.database import delete_old_predictions



def schedule_cleanup():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=delete_old_predictions,
        trigger="interval",
        days=1,
        id="cleanup_old_predictions",
        replace_existing=True
    )
    scheduler.start()
