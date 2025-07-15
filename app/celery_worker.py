from app.celery_utils import celery_app

# Auto-discover tasks in `app.tasks`
celery_app.autodiscover_tasks(["app.tasks"])

if __name__ == "__main__":
    celery_app.start()
