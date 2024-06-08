# import countvectorizer from sklearn
from sklearn.feature_extraction.text import CountVectorizer

# create an instance of CountVectorizer
vectorizer = CountVectorizer()

# create a list of text data
text_data = ["I am learning NLP. It is very interesting and exciting.",
             "I am learning python. It is also interesting and exciting.",
             "I am learning statistics, it is useful for data science.",
             "I am learning machine learning. It is very interesting and exciting"]

# fit the data and transform it into a vector
X = vectorizer.fit_transform(text_data)

# print the feature names
print(vectorizer.get_feature_names_out())

# print the vector
print(X.toarray())
