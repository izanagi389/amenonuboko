import requests

from app.lib.text.shape import TextShapping


def getData(url, api_key, columns):
    headers = {'X-API-KEY': api_key}
    contents = []
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()

        data = res.json()
        t = TextShapping()
        for content in data["contents"]:
            content_list = []
            for column in columns:
                if isinstance(content[column], list):
                    str = ""
                    for l in content[column]:
                        if not l.get('content') == None:
                            str += l["content"]

                    str = TextShapping.remove_html_tag(str)
                    str = TextShapping.remove_url(str)
                    # str = TextShapping().remove_stop_words(str)
                    content_list.append(str)
                else:
                    content_list.append(content[column])

            contents.append(content_list)

    except requests.exceptions.RequestException as e:
        print("エラー : ", e)
        return [str(e)]

    return contents
