from lib.html.tag_removing import *


def create_content_list(data):
    content_list = []
    for contents in data["contents"]:
        text = ""
        if "blogContent" in contents and contents['blogContent'] != None:
            for c in contents['blogContent']:
                if "content" in c:
                    text += remove_tags(c["content"])
        content_list.append(
            [contents["id"], contents["title"], ''.join(text.split())])

    return content_list
