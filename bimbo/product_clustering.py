from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.cluster import KMeans

vocab_size = 10000 # maximum number of words to keep (keeps the most frequent)
maxlen = 200 # maximum number of words in a review

class ProductClustering:
    def __int__(self):
        pass

    def preprocess(self,text):
        tokenizer = Tokenizer(num_words=maxlen)
        tokenizer.fit_on_texts(list(producto_tabla_df["ProcessedText"].values))
        X_train = tokenizer.texts_to_sequences(producto_tabla_df["ProcessedText"] .values)
        x_train = sequence.pad_sequences(X_train, maxlen = maxlen)
        kmeans = KMeans(n_clusters=30)
        kmeans.fit(x_train)
        y_kmeans = kmeans.predict(x_train)
