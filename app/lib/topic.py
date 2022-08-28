import pandas as pd
import app.lib.contents.microcms as m
from app.lib.algorithm.lda import LDA
from app.lib.text.shape import TextShapping
import app.lib.morphology.sudachi as morphology
from app.db.sqlalchemy.crud import crud
import re


class topic():

    def __init__(self):
        return

    def start(self, content_list :list):

        df_contents = pd.DataFrame(content_list, columns=["id", "title", "blogContent"])
        df_contents["blogContent"] = df_contents["blogContent"].apply(lambda x: TextShapping.remove_unnecessary_text(x))

        docs = list(df_contents["blogContent"])

        docs = morphology.Tokenize(docs)
        docs = TextShapping.remove_numbers(docs)
        docs = TextShapping.remove_one_word(docs)
        
        docs = LDA.bigram2docs(docs)
        dictionary = LDA.get_dictionary(docs)
        corpus = LDA.get_corpus(dictionary, docs)

        print("START Load LDA MODEL")
        model = LDA.load(dictionary, corpus)
        print("FINISH Load LDA MODEL")

        print("START LDA")
        topics = LDA.topics(model, corpus)
        print("FINISH LDA")

        pattern = re.compile("|".join(topics))

        df_contents["corpus"] = df_contents["blogContent"].apply(lambda x :",".join(list(set(re.findall(pattern, x)))) if bool(pattern.search(x)) else "")
        df_contents = df_contents.drop('blogContent', axis=1)
        df_contents = df_contents.drop('title', axis=1)
        crud().add_data_for_df(df_contents, "topic_corpus")