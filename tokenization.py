from nltk.tokenize import word_tokenize

text = "I am learning NLP. It is very interesting and exciting. It is also challenging."
tokens = word_tokenize(text)

print(tokens)
# ['I', 'am', 'learning', 'NLP', '.', 'It', 'is', 'very', 'interesting', 'and', 'exciting', '.', 'It', 'is', 'also', 'challenging', '.']