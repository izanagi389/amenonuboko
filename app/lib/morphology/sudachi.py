from sudachipy import tokenizer
from sudachipy import dictionary


def Tokenize(docs, hinshi_list=["名詞", "", "一般"]):

    tokenizer_obj = dictionary.Dictionary(dict_type="full").create()

    # Split the documents into tokens.
    mode = tokenizer.Tokenizer.SplitMode.C
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = [m.surface() for m in tokenizer_obj.tokenize(docs[idx], mode)
                     if check_for_hinshi(m.part_of_speech(), hinshi_list)]  # Split into words.

    return docs


def check_for_hinshi(part_of_speech, hinshi):

    part_of_speech_list = list(part_of_speech)

    flag = 0

    for i, h in enumerate(hinshi):

        if h == "":
            flag += 1
            continue
        if not part_of_speech_list[i] == h:
            return False

    if len(hinshi) == flag:
        return False
    else:
        return True
