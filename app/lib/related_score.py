import pandas as pd
import itertools
import numpy as np

import scipy.spatial

from app.lib.algorithm.sentence_bert import SentenceBertJapanese
from app.db.sqlalchemy.crud import crud


class create_score():

    def __init__(self):
        return

    def start(self, content_list: list):

        df_contents = pd.DataFrame(content_list, columns=[
                                   "id", "title", "blogContent"])

        df_new = pd.DataFrame(list(itertools.product(
            df_contents["id"], df_contents["id"])), columns=["id", "relate_id"])
        df_new["title"] = df_new.apply(
            lambda x: df_contents[df_contents["id"] == x["id"]]["title"].iloc[-1], axis=1)
        df_new["relate_title"] = df_new.apply(
            lambda x: df_contents[df_contents["id"] == x["relate_id"]]["title"].iloc[-1], axis=1)

        model = SentenceBertJapanese(
            "sonoisa/sentence-bert-base-ja-mean-tokens-v2")
        sentences = list(df_contents["title"])

        sentence_vectors = model.encode(sentences)

        queries = sentences
        # query_embeddings = model.encode(queries).numpy()
        query_embeddings = sentence_vectors.numpy()

        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist(
                [query_embedding], sentence_vectors, metric="cosine")[0]
            results = zip(range(len(distances)), distances)

            for idx, distance in results:
                df_new.loc[(df_new["title"] == query) & (
                    df_new["relate_title"] == sentences[idx]), ["bert_cos_distance"]] = distance / 2

        df_new = df_new.drop('title', axis=1)

        print(df_new)
        crud().add_data_for_df(df_new, "related_data_v2")
        print("h")

    def get_scores(self, num: int, search_id: str):

        df = crud().get_relate_score_for_df(search_id)
        return df[["relate_id", "relate_title", "bert_cos_distance"]].rename(columns={'relate_id': 'id', 'relate_title': 'title', 'bert_cos_distance': 'score'}).sort_values("score")[1:num+1].to_dict(orient='records')
