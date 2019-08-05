from collections import Counter, defaultdict
import re
import math


def text2words(text):
    words = re.split(' ', text)
    return words


class Doc_Categorization_NB:
    def __init__(self):
        self.Pc = {}  # Prior class prob P(C)
        self.defaultPWC = {}  # default probabilities of words not present in category's text corpus
        self.words_dict = defaultdict(defaultdict)  # {word: {cat: P(word|cat)} }

    def fit(self, text_corpus, labels, alpha = 1.0 ):
        '''
        :param text_collection: dict {class : [texts]}
        :return:
        '''

        total_texts = sum(len(texts) for texts in text_corpus)
        total_cats = len(labels)
        for cat, cat_texts in zip(labels, text_corpus):
            # Prior class prob P(C)
            self.Pc[cat] = len(cat_texts) / total_texts

            total_words_in_cat = 0
            words_counter = Counter()
            for text in cat_texts:
                words = text2words(text)
                total_words_in_cat += len(words)
                words_counter.update(words)

            words_in_cat_dict = len(words_counter.keys())
            self.defaultPWC[cat] = alpha / (total_words_in_cat + alpha * words_in_cat_dict)

            for word, word_count in words_counter.items():
                # P(W|C) using Laplace estimator
                PWC = (word_count + alpha)/(total_words_in_cat + alpha*words_in_cat_dict)
                PWC_dict = self.words_dict[word]
                PWC_dict[cat] = PWC


    def predict(self, text):
        words = text2words(text)
        log_prob = {cat: math.log(PC) for cat, PC in self.Pc.items()}
        for word in words:
            if word in self.words_dict.keys():
                for cat in log_prob.keys():
                    if cat in self.words_dict[word]:
                        PWC = self.words_dict[word][cat]
                    else:
                        PWC = self.defaultPWC[cat]
                    logPWC = math.log(PWC)
                    log_prob[cat] += logPWC

        cat_max, p_max = None, 0.0
        for cat, log_p in log_prob.items():
            p = math.exp(log_p)
            if p > p_max:
                cat_max = cat
                p_max = p

        return cat_max

text0 = "A AA, AAA B A AAA AA BB A AAA A AA"
words0 = text2words(text0)

labels = ['cat_A', 'cat_B']
text_corpus = [
    # A texts
    ["A AA B A AAA AA BB A AAA A AA",
     "B AAA AA A A A AA BB",
     "AA AAA AA A B AAA AA A",
     "A A AA AAA BB"],
    # B texts
    ["BBB B A BB B BBB BB AA",
     "BB, B B B A BB BB BBB",
     "B BBB BB B AA B BBB"]
]

textA = "CC AA A B AA AAA AAA AAA"
textB = "BB dd AA BB BBB B B B"

model = Doc_Categorization_NB()
model.fit(text_corpus, labels)

resA = model.predict(textA)
resB = model.predict(textB)

pass


