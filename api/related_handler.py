from flask import Blueprint
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import json
import setup
from lib.mecab.tokenize import *
from lib.json.json_creating import *
from lib.html.tag_removing import *
from lib.model_creating import *


related_doc_list = Blueprint(
    'related_doc_list', __name__, url_prefix='/related_doc_list')


@related_doc_list.route('/<serach_id>', methods=['GET'])
def related(serach_id):

    serach_id = remove_tags(serach_id)

    if (serach_id == None):
        return json.dumps(["パラメータがありません！"], indent=2, ensure_ascii=False)

    model_file_path = get_model_path()

    df = pd.read_csv(setup.CSV_FILE_PATH)
    model = Doc2Vec.load(model_file_path)

    most_similar_id_list = model.docvecs.most_similar(
        serach_id, topn=setup.NUM)

    return create_json_data(df, most_similar_id_list)
