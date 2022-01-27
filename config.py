
from os.path import join, dirname
from dotenv import load_dotenv
import os
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

MICROCMS_API_KEY = os.environ.get('MICROCMS_API_KEY')
MICROCMS_URL = os.environ.get('MICROCMS_URL')
LIMIT = os.environ.get('LIMIT')

BERT_MODEL_NAME = os.environ.get('BERT_MODEL_NAME')
BERT_FINETUNING_MODEL_NAME = os.environ.get('BERT_FINETUNING_MODEL_NAME')
CL_TOHOKU_MODEL_NAME = os.environ.get('CL_TOHOKU_MODEL_NAME')
COLUMNS = ["id", "title"]

MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_HOST = 'db'
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE')
SITE_ROOT_URL = os.environ.get('SITE_ROOT_URL')
ORIGINS = [
    "http://localhost:3000",
    SITE_ROOT_URL,
]
