
from os.path import join, dirname
from dotenv import load_dotenv
import os
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

import os
MICROCMS_API_KEY = os.getenv('MICROCMS_API_KEY')
MICROCMS_URL = os.getenv('MICROCMS_URL')
LIMIT = os.getenv('LIMIT')

BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME')
BERT_FINETUNING_MODEL_NAME = os.getenv('BERT_FINETUNING_MODEL_NAME')
CL_TOHOKU_MODEL_NAME = os.getenv('CL_TOHOKU_MODEL_NAME')
COLUMNS = ["id", "title"]