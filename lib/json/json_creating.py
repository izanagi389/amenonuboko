import json
# import pandas as pd
# import setup


def create_json_data(df, most_similar_id_list):
    result_list = []
    for id, score in most_similar_id_list:
        result = {}
        result["id"] = id
        result["score"] = score
        result["title"] = df.loc[df["id"] == id]["title"].item()
        result_list.append(result)
    return json.dumps(result_list, indent=2, ensure_ascii=False)
