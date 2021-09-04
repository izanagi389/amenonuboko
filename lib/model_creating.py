from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
from datetime import datetime
import setup
import glob
from lib.json_operation import *
from lib.data_frame_operation import *
from lib.model_creating import *
from lib.content_list_creating import *
from lib.text.normalization import *
from lib.mecab.tokenize import *


def doc2vec_model_create():

    url = setup.MICROCMS_API_URL
    api_key = setup.MICROCMS_API_KEY
    model_file_path = get_model_path()
    dt_now = datetime.now().strftime(setup.DATE_FORMAT)
    if (model_file_path != None):
        os.remove(model_file_path)

    json = get_json_data(url, api_key)
    content_list = create_content_list(json)
    df = create_data_frame(content_list)

    sentences = {}
    for d in df["id"]:
        text = normalization(df.loc[df["id"] == d]["text"].item())
        text_list = tokenize(text)
        # model_id = d + "{span}" + df.loc[df["id"] == d]["title"].item()
        sentences[d] = text_list

    documents = [TaggedDocument(doc, [i]) for i, doc in sentences.items()]
    model = Doc2Vec(documents, vector_size=setup.VECTOR_SIZE, dm=setup.DM, window=setup.WINDOW, alpha=setup.ALPHA,
                    min_alpha=setup.MIN_ALPHA, min_count=setup.MIN_COUNT, sample=setup.SAMPLE, epochs=setup.EPOCHS)
    model.save(setup.MODEL_DIR_PATH + dt_now + "-" + setup.MODEL_NAME)


def get_model_path():
    model_file_path_list = glob.glob(setup.MODEL_DIR_PATH + '*.model')

    if model_file_path_list:
        model_file_path = model_file_path_list[0]
    else:
        model_file_path = None

    return model_file_path
