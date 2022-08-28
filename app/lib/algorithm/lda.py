from gensim.models import LdaModel, Phrases
from gensim.corpora import Dictionary


class LDA():

    def __init__(self):
        return

    def load(dictionary, corpus, num_topics=5, chunksize=2000, passes=10, iterations=400, eval_every=None):
        # Make an index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

        return model

    # Compute bigrams.
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    def bigram2docs(docs: list,  min_count=20) -> list:
        bigram = Phrases(docs, min_count)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)

        return docs

    def get_dictionary(docs):
        # Remove rare and common tokens.
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(docs)

        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        dictionary.filter_extremes(no_below=5, no_above=0.5)

        return dictionary



    def get_corpus(dictionary, docs):
        # Bag-of-words representation of the documents.
        corpus = [dictionary.doc2bow(doc) for doc in docs]

        return corpus


    def topics(model, corpus) -> any:
        top_topics = model.top_topics(corpus)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        # print('Average topic coherence: %.4f.' % avg_topic_coherence)

        topic_list = []

        for topic in top_topics:
            for tt in topic:
                if isinstance(tt, list):
                    for t in tt:
                        topic_list.append(t[1])

        topic_list = list(set(topic_list))

        return topic_list
