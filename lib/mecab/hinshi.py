import setup


def remove_hinshi(row, word):
    hinshi = row.split("\t")[1].split(",")
    if hinshi[0] in setup.TOP_HINSHI:
        if not hinshi[1] in setup.SUB_HINSHI:
            if not len(word) == 1:
                return word
