import requests
import json


def get_json_data(url, api_key):
    headers = {'X-API-KEY': api_key}
    res = requests.get(url, headers=headers)
    data = res.json()
    return data


def create_json_data(df, most_similar_id_list):
    result_list = []
    for id, score in most_similar_id_list:
        for d in df["id"]:
            result = {}
            if d == id:
                result["id"] = id
                result["score"] = score
                result["title"] = df.loc[df["id"] == id]["title"].item()
                result_list.append(result)
    return json.dumps(result_list, indent=2, ensure_ascii=False)
