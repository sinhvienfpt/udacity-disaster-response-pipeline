# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
text = "I am learning NLP. It is very interesting and exciting. It is also challenging."

# tokenize the text
tokens = word_tokenize(text)
print(tokens)
for token in tokens:
    if token != ps.stem(token): # learning : learn
        print(token + " ⇒ " + ps.stem(token))


