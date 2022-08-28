import app.lib.contents.microcms as m
from app.lib.topic import topic
from app.lib.related_score import create_score

import datetime
import config

def init():

    print("Scheduling Start! at {}".format(datetime.datetime.today()))

    url = config.MICROCMS_URL + "?limit=" + config.LIMIT
    api_key = config.MICROCMS_API_KEY
    columns = ["id", "title", "blogContent"]

    content_list = m.getData(url, api_key, columns)

    # トピックコーパスの作成
    print("Creating Topic Corpus")
    topic().start(content_list)
    print("Created Topic Corpus")

    # 関連記事のスコア作成（v2）
    print("Creating Related Score")
    create_score().start(content_list)
    print("Created Related Score")
