from flask import Flask
from app.celery import make_celery
from config.logging_config import setup_logging

logger = setup_logging()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.config.Config')

    celery = make_celery(app)

    from app.routes import api
    app.register_blueprint(api)

    return app, celery
