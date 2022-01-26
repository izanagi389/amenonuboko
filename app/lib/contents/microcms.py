
import requests


def getData(url, api_key, columns):
    headers = {'X-API-KEY': api_key}
    contents = []
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()

        data = res.json()
        for content in data["contents"]:
            list = []
            for column in columns:
                list.append(content[column])

            contents.append(list)

    except requests.exceptions.RequestException as e:
        print("エラー : ", e)
        return [str(e)]

    return contents
