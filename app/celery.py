from celery import Celery
from tasks.pdf_processing import process_pdf_async  # Ensure this import is correct

celery = None

def make_celery(app):
    global celery
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    
    celery.conf.update(app.config)
    celery.autodiscover_tasks(['tasks'])

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
