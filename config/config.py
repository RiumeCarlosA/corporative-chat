import os

class Config:
    # Configurações do Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'minha_chave_secreta')

    # Configurações do Celery e Redis
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
