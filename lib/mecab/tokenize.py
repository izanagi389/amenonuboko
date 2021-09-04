import MeCab
import setup


def tokenize(text):

    tagger = MeCab.Tagger(setup.MODEL_DIR_PATH)

    key = tagger.parse(text)

    corpus = []
    for row in key.split("\n"):
        word = row.split("\t")[0]
        if word == "EOS":
            break
        else:
            corpus.append(word)
            # remove_hinshi(row, word)

    return corpus
