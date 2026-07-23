import numpy as np


class SimpleWord2Vec:
    """
    Simple Word2Vec implementation using NumPy.
    Skip-Gram model with basic gradient descent.
    """

    def __init__(
        self,
        vector_size=50,
        window_size=2,
        learning_rate=0.01,
        epochs=100
    ):
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.word_to_index = {}
        self.index_to_word = {}

        self.W = None
        self.W_out = None


    def build_vocab(self, sentences):

        words = []

        for sentence in sentences:
            words.extend(sentence.split())


        unique_words = list(set(words))


        self.word_to_index = {
            word: i
            for i, word in enumerate(unique_words)
        }


        self.index_to_word = {
            i: word
            for word, i in self.word_to_index.items()
        }


        vocab_size = len(unique_words)


        self.W = np.random.uniform(
            -0.5,
            0.5,
            (vocab_size, self.vector_size)
        )


        self.W_out = np.random.uniform(
            -0.5,
            0.5,
            (vocab_size, self.vector_size)
        )


    def create_training_pairs(self, sentences):

        pairs = []

        for sentence in sentences:

            words = sentence.split()


            for i, word in enumerate(words):

                center = self.word_to_index[word]


                start = max(
                    0,
                    i - self.window_size
                )

                end = min(
                    len(words),
                    i + self.window_size + 1
                )


                for j in range(start, end):

                    if i != j:

                        context = self.word_to_index[
                            words[j]
                        ]

                        pairs.append(
                            (center, context)
                        )

        return pairs


    def train(self, sentences):

        self.build_vocab(sentences)


        pairs = self.create_training_pairs(
            sentences
        )


        for epoch in range(self.epochs):

            loss = 0


            for center, context in pairs:

                hidden = self.W[center]


                score = np.dot(
                    self.W_out,
                    hidden
                )


                exp_score = np.exp(
                    score - np.max(score)
                )

                probs = exp_score / np.sum(exp_score)


                error = probs.copy()

                error[context] -= 1


                self.W_out -= (
                    self.learning_rate *
                    np.outer(
                        error,
                        hidden
                    )
                )


                self.W[center] -= (
                    self.learning_rate *
                    np.dot(
                        error,
                        self.W_out
                    )
                )


                loss -= np.log(
                    probs[context] + 1e-9
                )


            if epoch % 20 == 0:
                print(
                    f"Word2Vec Epoch {epoch}, Loss: {loss:.4f}"
                )


    def get_vector(self, word):

        if word in self.word_to_index:

            return self.W[
                self.word_to_index[word]
            ]

        return np.zeros(
            self.vector_size
        )


    def sentence_vector(self, sentence):

        vectors = []

        for word in sentence.split():

            vectors.append(
                self.get_vector(word)
            )


        if len(vectors) == 0:
            return np.zeros(
                self.vector_size
            )


        return np.mean(
            vectors,
            axis=0
        )


def build_word2vec_embeddings(sentences):

    model = SimpleWord2Vec(
        vector_size=50,
        window_size=2,
        learning_rate=0.01,
        epochs=100
    )


    model.train(sentences)


    embeddings = np.array(
        [
            model.sentence_vector(sentence)
            for sentence in sentences
        ]
    )


    return embeddings, model