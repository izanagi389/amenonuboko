import pandas as pd
import app.math as transformers_math_lib
import app.touch_model as transformers_model_lib
import app.microcms as m
import config
import itertools
import numpy as np
from app.database import SessionLocal, engine


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_related_titles(search_id: str):
    query = "select id, title, `{}` from related_text.contents".format(search_id)
    df = pd.read_sql(query, con=engine)

    # return df.sort_values(search_id, ascending=False)[:5].rename(columns={search_id: 'score'}).to_json(orient='records', force_ascii=False)
    return df.sort_values(search_id, ascending=False)[:5].rename(columns={search_id: 'score'}).to_dict(orient='records')

def init_data():
    url = config.MICROCMS_URL + "?limit=" + config.LIMIT
    api_key = config.MICROCMS_API_KEY
    columns = config.COLUMNS

    content_list = m.getData(url, api_key, columns)

    df = pd.DataFrame(content_list, columns=config.COLUMNS)

    tknz, model = transformers_model_lib.get_model()

    tm = transformers_math_lib.Math(tknz, model)

    print("Strat sentence_vector")
    df["score"] = df.apply(lambda x: tm.sentence_vector(x["title"]), axis=1)
    columns_add = df["id"].values.tolist()
    for newcol in columns_add:
        if newcol in df.columns:
            df[newcol] = 0.0

    array = list(itertools.permutations(df["id"].values.tolist(), 2))
    for l in array:
        v1 = tm.sentence_vector(df[df['id'] == l[0]]["title"].iloc[-1])
        v2 = tm.sentence_vector(df[df['id'] == l[1]]["title"].iloc[-1])
        i = df.reset_index().query('id == @l[0]').index[0]

        df.at[i, l[1]] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    df.to_sql('contents', engine, index=False, if_exists='replace')

