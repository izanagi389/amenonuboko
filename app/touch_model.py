

from transformers import BertJapaneseTokenizer, BertModel
import config


def initialize_cl_tohoku_load():
    print("Load Model")
    tknz = BertJapaneseTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tknz.save_pretrained(config.CL_TOHOKU_MODEL_NAME)
    print("Saved Model")


def get_model():
    # 日本語トークナイザ
    tknz = BertJapaneseTokenizer.from_pretrained(config.CL_TOHOKU_MODEL_NAME)
    model = BertModel.from_pretrained(config.BERT_FINETUNING_MODEL_NAME)

    return tknz, model
