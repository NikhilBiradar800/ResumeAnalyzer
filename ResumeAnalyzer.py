# Importing necessary libraries
import pandas as pd
import nltk
from nltk.util import ngrams
from gensim import corpora, models
from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
import math

##################### Start of the Analysis #####################

# Loading resume data from a CSV file given
resumes = pd.read_csv('resumes.csv', encoding='ISO-8859-1')   # We can add any resume file in CSV form

# Initializing NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Initializing NLTK punkt tokenizer
nltk.download('punkt')

# Cleaning text
def clean_text(text):
    # Check if 'resume_text' is NaN or empty
    if pd.isna(text) or not text.strip():
        return ''

    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Generating bi-grams and tri-grams
def generate_ngrams(text, n=2):
    words = text.split()
    n_grams = list(ngrams(words, n))
    return ['_'.join(gram) for gram in n_grams]
    


# Performing topic detection
def perform_topic_detection(text):
    tokens = text.split()
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Checking if 'resume_text' is empty or None
    if not tokens:
        return []

    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)
    topics = lda_model.print_topics()
    return topics

# Word cloud generation
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Statistics calculation
def calculate_statistics(text):
    num_words = len(text.split())
    unique_words = len(set(text.split()))
    entropy = calculate_entropy(text)
    return num_words, unique_words, entropy

# Calculating entropy
def calculate_entropy(text):
    word_list = text.split()
    word_count = len(word_list)
    word_freq = {}
    for word in word_list:
        word_freq[word] = word_freq.get(word, 0) + 1

    entropy = 0.0
    for word in word_freq:
        probability = word_freq[word] / word_count
        entropy -= probability * math.log(probability, 2)

    return entropy

# Network generation
def generate_network(text):
    G = nx.Graph()
    words = text.split()
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i+1])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

# Creating a report document to write the findings
with open('report.txt', 'w') as report:
    for index, row in resumes.iterrows():
        text = clean_text(row['resume_text']) 

        # Check if 'resume_text' is empty or None
        if not text:
            continue  # Skip empty or missing text

        # Bi-grams and Tri-grams
        bi_grams = generate_ngrams(text, n=2)
        tri_grams = generate_ngrams(text, n=3)
        print(f"Bigrams for Resume {index + 1}: {bi_grams}")
        print(f"Trigrams for Resume {index + 1}: {tri_grams}")

        # Topic Detection
        topics = perform_topic_detection(text)
        print(f"Topics for Resume {index + 1}: {topics}")

        # Word Cloud
        print(f"Generating Word Cloud for Resume {index + 1}...")
        generate_wordcloud(text)

        # Statistics
        num_words, unique_words, entropy = calculate_statistics(text)
        print(f"Statistics for Resume {index + 1}:")
        print(f"Number of Words: {num_words}")
        print(f"Total Unique Words: {unique_words}")
        print(f"Entropy: {entropy}")

        # Network Generation
        print(f"Generating Network for Resume {index + 1}...")
        generate_network(text)

        # Writing information in the report
        report.write(f"Resume {index + 1} Analysis:\n")
        report.write(f"Number of Words: {num_words}\n")
        report.write(f"Total Unique Words: {unique_words}\n")
        report.write(f"Topics: {topics}\n")
        report.write(f"Entropy: {entropy}\n")

