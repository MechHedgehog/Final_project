import gensim.models
import string
import codecs
import glob
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def tokin(text):
    tokens = word_tokenize(text, language="russian")
    tokens = [word.lower() for word in tokens if(word not in string.punctuation)]
    return tokens


kot = ""
texts = glob.glob("_mf*.txt")
for i in texts:
    inp = codecs.open(i, mode="r", encoding="utf-8")
    kot += inp.read()


sentences = [tokin(sent) for sent in sent_tokenize(kot, language="russian")]
print(len(sentences))
model = gensim.models.Word2Vec(sentences, size=256, window=7, min_count=5, workers=8)


model.save('8book.model')
print('saved')
print(len(model.wv.vocab))