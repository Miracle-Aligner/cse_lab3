import pickle
import string

from pymongo import MongoClient
from sklearn.decomposition import PCA, TruncatedSVD
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import MDS

from tqdm import tqdm

# initialize mongoDb url
MONGO_URL = 'mongodb://Kekichi:PythonCerf1@ds245170.mlab.com:45170/pythonlab'

# Functions

def mongo_data():
    client = MongoClient(MONGO_URL)
    print("Receiving data from Mongo...")
    messages = []
    for d in tqdm(client.cselab.data.find()):
        messages.append(d['message'])
    with open("messages", "wb") as file:
        pickle.dump(messages, file)
    print("\nReceived data from Mongo...")
    return messages;

def process(messages):
    tokenizer = RegexpTokenizer(r"\w+")
    lemmatizer = WordNetLemmatizer()
    print("Processing words...")
    processed = []
    for m in tqdm(messages):
        tokens = [t for t in tokenizer.tokenize(str.lower(m)) if t not in stopwords.words("english")]
        if len(tokens) > 0:
            processed.append([lemmatizer.lemmatize(t) for t in tokens])

    with open("processed", "wb") as file:
        pickle.dump(processed, file)

    print("Processed...")
    return processed;

def plot_word_cloud(processed):
    text = " ".join(y for x in processed for y in x)
    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def plot(X):
    clusters = 8
    kmeans = KMeans(n_clusters=clusters)
    kmeans_result = kmeans.fit(X)
    cluster_labels = kmeans.labels_.tolist()
    # Вывод результатов
    # pca = PCA(n_components=2).fit(X)
    # data2D = pca.transform(X)
    data2D = TruncatedSVD().fit_transform(X)
    plt.scatter(data2D[:, 0], data2D[:, 1], c=kmeans.labels_)
    # mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    # pos = mds.fit_transform(kmeans_result)
    # plt.scatter(pos[:, 0], pos[:, 1], marker="x", c=kmeans.labels_)
    plt.show()

def search(X):
    print("Searching for the best cluster size...")
    Nc = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in tqdm(Nc)]
    score = [kmean.fit(X).score(X) for kmean in kmeans]
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    print("Found the best cluster size")


# MAIN LOGIC
# Receiving messages
messages = mongo_data();
# Preparing messages
processed = process(messages)
print(processed)
plot_word_cloud(processed)

# Processing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform([y for x in processed for y in x])
# Finding claster's size
search(X)
# Building result
plot(X)

