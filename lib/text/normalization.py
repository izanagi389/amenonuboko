import emoji
import re
import setup


def normalization(text):
    text = re.sub(
        r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", text)
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
    text = ''.join(char for char in text if char.isalnum())

    for sw in setup.SKIP_WORDS:
        text = re.sub(sw, "", text)

    return text
