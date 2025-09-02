from apscheduler.schedulers.background import BackgroundScheduler

def schedule_jobs():
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(lambda: weekly_reminders(), "cron", day_of_week="mon", hour=9, minute=0)
    scheduler.add_job(lambda: precutoff_reminders(), "cron", hour=9, minute=0)  # daily 9AM
    scheduler.start()

@app.on_event("startup")
def on_start():
    schedule_jobs()
