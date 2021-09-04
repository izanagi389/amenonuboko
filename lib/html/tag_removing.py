from bs4 import BeautifulSoup


def remove_tags(tag_text):
    soup = BeautifulSoup(tag_text, 'html.parser')

    return soup.text
