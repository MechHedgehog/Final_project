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


books_name = "mf*.txt"
model_name = "w2v.model"


texts = glob.glob(books_name)
model = gensim.models.Word2Vec.load(model_name)
print(len(model.wv.vocab))
for i in texts:
    inp = codecs.open(i, mode="r", encoding="utf-8")
    kot = inp.read()
    inp.close()
    sentences = [tokin(sent) for sent in sent_tokenize(kot, language="russian")]
    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print(len(model.wv.vocab))

model.save("8book_another.model")