import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


class Summarizer:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.wt = word_tokenize
        self.st = sent_tokenize

    def summarize(self, text, max_len=256):

        freq_table = self._word_tokenization(text)

        sentence_value = dict()
        sentences = self.st(text)
        for sentence in sentences:
            for word, freq in freq_table.items():
                if word in sentence.lower():
                    if sentence in sentence_value:
                        sentence_value[sentence] += freq
                    else:
                        sentence_value[sentence] = freq

        sentence_value = dict(sorted(sentence_value.items(), key=lambda item: item[1], reverse=True))

        output = ""
        output_len = 0

        for sentence, _ in sentence_value.items():
            sent_len = len(sentence.split())
            if output_len + sent_len <= max_len:
                output += sentence
                output_len += sent_len
            else:
                remain = max_len - output_len
                remain_sent = " ".join(sentence.split()[0:remain])
                output += remain_sent
                output_len += remain
                break

        return output

    def _word_tokenization(self, text):
        freq_table = {}
        words = self.wt(text)

        self.wt(text)
        for word in words:
            word = word.lower()
            if word in self.stop_words:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        return freq_table
