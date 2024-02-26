from keybert import KeyBERT


def get_keywords(doc, number_of_words=1, top_n=4):
    """Utility function to generate keywords for a piece of text."""
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, number_of_words),
        diversity=0.35,
        top_n=top_n,
    )
    return keywords


if __name__ == "__main__":
    doc = """
     Supervised learning is the machine learning task of learning a function that
     maps an input to an output based on example input-output pairs. It infers a
     function from labeled training data consisting of a set of training examples.
     In supervised learning, each example is a pair consisting of an input object
     (typically a vector) and a desired output value (also called the supervisory signal).
     A supervised learning algorithm analyzes the training data and produces an inferred function,
     which can be used for mapping new examples. An optimal scenario will allow for the
     algorithm to correctly determine the class labels for unseen instances. This requires
     the learning algorithm to generalize from the training data to unseen situations in a
     'reasonable' way (see inductive bias).
    """
    print(get_keywords(doc, number_of_words=1, top_n=5))
    # Output: [('banana', 0.8553), ('kittens', 0.8476)]
    # The output may vary depending on the model used.
    print(get_keywords(doc, number_of_words=2, top_n=3))
