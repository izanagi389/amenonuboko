
from os.path import join, dirname
from dotenv import load_dotenv
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATE_FORMAT = '%Y%m%d'

# microCMSセッティング
MICROCMS_API_KEY = os.environ.get("MICROCMS_API_KEY")
MICROCMS_API_URL = os.environ.get("MICROCMS_API_URL")

# 一時記事データ保存用保存用（空文字削除のため）
CSV_FILE_PATH = "static/csv/micro-cms-contents.csv"

# MECAB設定
MECAB_DIC_URL = os.environ.get("MECAB_DIC_URL")

TOP_HINSHI = ["名詞", "動詞", "固有名詞"]
SUB_HINSHI = ["自立", "非自立", "接尾", "副詞可能"]

# Doc2vec設定
MODEL_DIR_PATH = "static/model/"
MODEL_NAME = "doc2vec.model"
SKIP_WORDS = ["みなさんこんにちはイザナギです",
              "それでは今回はここで筆を置かせていただきます最後まで記事をご覧いただきありがとうございました"]

VECTOR_SIZE = 100
DM = 0
WINDOW = 15
ALPHA = .025
MIN_ALPHA = .025
MIN_COUNT = 1
SAMPLE = 1e-6
EPOCHS = 50

# 結果出力数
NUM = 4
