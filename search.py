# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Βιβλιοθήκες
import requests
from bs4 import BeautifulSoup
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
from nltk.stem import WordNetLemmatizer

# Ερώτημα 1
# a Επιλέξτε έναν ιστότοπο-στόχο ή ένα αποθετήριο ακαδημαϊκών εργασιών (π.χ. arXiv, PubMed ή αποθετήριο πανεπιστημίου). 
# β. Υλοποιήστε έναν web crawler σε Python (π.χ. με BeautifulSoup) για τη συλλογή 
# μεταδεδομένων ακαδημαϊκών εργασιών (τίτλος, συγγραφείς, περίληψη, ημερομηνία 
# δημοσίευσης κ.λπ.) από την επιλεγμένη πηγή.
# γ. Αποθηκεύστε τα δεδομένα που συλλέγονται σε δομημένη μορφή, όπως JSON ή CSV.

def scrape_polynoe():
    url = 'https://polynoe.lib.uniwa.gr/xmlui/browse?type=dateissued'
    html = requests.get(url)
    soup = BeautifulSoup(html.content, 'html.parser')
    descriptions = soup.find_all('div', class_='artifact-description')
    data = []

    for desc in descriptions:
        title = desc.find('h4', class_='artifact-title').text.strip()
        author = desc.find('span', class_='author h4').text.strip()
        date = desc.find('span', class_='date').text.strip()
        abstract = desc.find('div', class_='artifact-abstract').text.strip()
        data.append([title, author, date, abstract])

    with open('data.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Ερώτημα 2
# Κάντε προεπεξεργασία του κειμενικού περιεχομένου των ακαδημαϊκών εργασιών για την προετοιμασία τους για ευρετηρίαση και αναζήτηση. 
# Αυτό μπορεί να περιλαμβάνει εργασίες όπως tokenization, stemming/lemmatization και stop-word removal και αφαίρεση ειδικών χαρακτήρων 
# (removing special characters).

def preprocess_text():

    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer
    lemmatizer = WordNetLemmatizer()  # Define lemmatizer

    with open('data.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    processed_data = []
    for entry in data:
        title_tokens = word_tokenize(entry[0])
        title_tokens = [word.lower() for word in title_tokens if word.isalpha() and word.lower() not in stop_words]
        title_tokens = [stemmer.stem(word) for word in title_tokens]
        title_tokens = [lemmatizer.lemmatize(word) for word in title_tokens]

        author_tokens = word_tokenize(entry[1])
        author_tokens = [word.lower() for word in author_tokens if word.isalpha() and word.lower() not in stop_words]
        author_tokens = [stemmer.stem(word) for word in author_tokens]
        author_tokens = [lemmatizer.lemmatize(word) for word in author_tokens]

        abstract_tokens = word_tokenize(entry[3])
        abstract_tokens = [word.lower() for word in abstract_tokens if word.isalpha() and word.lower() not in stop_words]
        abstract_tokens = [stemmer.stem(word) for word in abstract_tokens]
        abstract_tokens = [lemmatizer.lemmatize(word) for word in abstract_tokens]

        processed_data.append({
            'title': title_tokens,
            'author': author_tokens,
            'date': entry[2],
            'abstract': abstract_tokens
        })

    with open('processed_data.json', 'w', encoding='utf8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

# Ερώτημα 3
# α. Δημιουργήστε μια ανεστραμμένη δομή δεδομένων ευρετηρίου (inverted index) για την αποτελεσματική αντιστοίχιση όρων
#  στα έγγραφα στα οποία εμφανίζονται. 
# β. Εφαρμόστε μια δομή δεδομένων για την αποθήκευση του ευρετηρίου.

def create_inverted_index():
    with open('processed_data.json', 'r', encoding='utf8') as f:
        data = json.load(f)
    inverted_index = defaultdict(set)
    for i, entry in enumerate(data):
        for word in entry['abstract']:
            inverted_index[word].add(i)
    inverted_index = {k: list(v) for k, v in inverted_index.items()}
    with open('inverted_index.json', 'w', encoding='utf8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=4)


# Ερώτημα 4
# α. Αναπτύξτε μια διεπαφή χρήστη για την αναζήτηση ακαδημαϊκών εργασιών χρησιμοποιώντας την Python (π.χ. μια διεπαφή γραμμής εντολών ή μια απλή διεπαφή ιστού). 

def search(search_query):
    print("Please choose an algorithm:")
    print("1. Boolean Retrieval")
    print("2. Vector Space Model")
    print("3. Okapi BM25")
    choice = int(input("Enter your choice (1-3): "))

    if choice == 1:
        print(boolean_retrieval(search_query))
    elif choice == 2:
        print(ranking(search_query,'vectorspacemodel'))
    elif choice == 3:
        print(ranking(search_query,'okapibm25'))
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")
        return

# β. Υλοποιήστε πολλαπλούς (τουλάχιστον 3) αλγόριθμους ανάκτησης, όπως Boolean retrieval, 
# Vector Space Model (VSM) και Probabilistic retrieval models (π.χ. Okapi BM25) για να 
# ανακτήσετε σχετικές εργασίες με βάση τα ερωτήματα των χρηστών. Ο χρήστης θα μπορεί 
# να επιλέγει τον αλγόριθμο ανάκτησης.

def boolean_retrieval(query):
    query = query_processing(query)

    # Load the inverted index from the JSON file
    with open('inverted_index.json', 'r', encoding='utf8') as f:
        inverted_index = json.load(f)

    # Initialize the set of documents
    docs = set(inverted_index.get(query[0], []))

    # Apply Boolean operators
    for i in range(1, len(query), 2):
        operator = query[i]
        word = query[i+1]

        if operator.lower() == 'and':
            docs &= set(inverted_index.get(word, []))
        elif operator.lower() == 'or':
            docs |= set(inverted_index.get(word, []))
        elif operator.lower() == 'not':
            docs -= set(inverted_index.get(word, []))

    return list(docs)

def vector_space_model(query):
    # Load preprocessed documents from JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    # Tokenize the query
    tokenized_query = word_tokenize(query.lower())

    # Calculate TF-IDF
    # Convert tokenized documents to text
    preprocessed_documents = [' '.join(doc['title'] + doc['author'] + doc['abstract'] + [doc['date']]) for doc in documents]  # Combine all fields
    preprocessed_query = ' '.join(tokenized_query)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

    # Transform the query into a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

    # Rank documents by similarity
    results = [(documents[i], cosine_similarities[0][i]) for i in range(len(documents))]
    results.sort(key=lambda x: x[1], reverse=True)

    # Print the top 5 ranked documents
    # for doc, similarity in results[:5]:  
    #     print(f"Similarity: {similarity:.2f}\nTitle: {' '.join(doc['title'])}\nAuthor: {' '.join(doc['author'])}\nDate: {doc['date']}\nAbstract: {' '.join(doc['abstract'])}\n")  # Print all fields
    return results

def okapibm25(query):
    # Load preprocessed documents from JSON file
    with open('processed_data.json', 'r', encoding='utf8') as f:
        documents = json.load(f)

    # Tokenize the query
    tokenized_query = query.split(" ")

    # Convert tokenized documents to text
    preprocessed_documents = [' '.join(doc['title'] + doc['author'] + doc['abstract'] + [doc['date']]) for doc in documents]  # Combine all fields

    # Initialize BM25Okapi model
    bm25 = BM25Okapi([doc.split(" ") for doc in preprocessed_documents])

    # Get scores for each document
    doc_scores = bm25.get_scores(tokenized_query)

    # Get the indices of the top documents
    top_indices = bm25.get_top_n(tokenized_query, range(len(preprocessed_documents)), n=5)

    # # Print the details of the top documents
    # for index in top_indices:
    #     print(f"Similarity Score: {doc_scores[index]}")
    #     print(f"Title: {documents[index]['title']}")
    #     print(f"Author: {documents[index]['author']}")
    #     print(f"Abstract: {documents[index]['abstract']}")
    #     print(f"Date: {documents[index]['date']}")
    #     print("\n")
    results = [(documents[i], doc_scores[i]) for i in top_indices]
    results.sort(key=lambda x: x[1], reverse=True)

    return results


# γ. Επιτρέψτε στους χρήστες να φιλτράρουν τα αποτελέσματα αναζήτησης με διάφορα 
# κριτήρια, όπως η ημερομηνία δημοσίευσης ή ο συγγραφέας.

# def filter_results(criteria, value):
#     # Άνοιγμα του αρχείου με τα επεξεργασμένα δεδομένα
#     with open('processed_data.json', 'r', encoding='utf8') as f:
#         data = json.load(f)

#     # Δημιουργία μιας λίστας με τα έγγραφα που πληρούν το κριτήριο
#     filtered_data = [doc for doc in data if doc.get(criteria) == value]
#         print(filtered_data)

#     # Επιστροφή της λίστας με τα φιλτραρισμένα δεδομένα
#     return filtered_data

# Επεξεργασία ερωτήματος (Query Processing): Αναπτύξτε ένα module επεξεργασίας 
# ερωτημάτων που θα προεπεξεργάζεται τα ερωτήματα που λαμβάνει από τον χρήστη, τα αναλύει 
# και ανακτά σχετικά έγγραφα χρησιμοποιώντας το ανεστραμμένο ευρετήριο. Μπορείτε να 
# χρησιμοποιήσετε απλά ερωτήματα βάσει λέξεων (όρων). Οι χρήστες θα πρέπει να μπορούν να 
# αναζητούν έγγραφα χρησιμοποιώντας μία ή περισσότερες λέξεις. Το module θα λαμβάνει 
# ερωτήματα χρηστών τα οποία τα γίνονται tokenized και θα εκτελεί λειτουργίες Boolean (AND, OR
# και NOT). 

def query_processing(query):

    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer
    lemmatizer = WordNetLemmatizer()  # Define lemmatizer

    query_tokens = word_tokenize(query)
    query_tokens = [word.lower() for word in query_tokens if word.isalpha() and word.lower() not in stop_words]
    query_tokens = [stemmer.stem(word) for word in query_tokens]
    query_tokens = [lemmatizer.lemmatize(word) for word in query_tokens]
    
    return query_tokens


# Κατάταξη αποτελεσμάτων (Ranking): Εφαρμόστε έναν βασικό αλγόριθμο κατάταξης. Μπορείτε 
# να ξεκινήσετε με έναν απλό αλγόριθμο κατάταξης TF-IDF (Term Frequency-Inverse Document
# Frequency) και αργότερα μπορείτε να συμπεριλάβετε πιο προηγμένες τεχνικές κατάταξης. 
# Ταξινομήστε και παρουσιάστε τα αποτελέσματα αναζήτησης σε φιλική προς το χρήστη μορφή.

def ranking(query, ranking_algorithm):
    if ranking_algorithm == 'vectorspacemodel':
        results = vector_space_model(query)
    elif ranking_algorithm == 'okapibm25':
        results = okapibm25(query)
    else:
        raise ValueError("Unsupported ranking algorithm")
    for doc, similarity in results:
        print(f"Similarity: {similarity:.2f}\nTitle: {' '.join(doc['title'])}\nAuthor: {' '.join(doc['author'])}\nDate: {doc['date']}\nAbstract: {' '.join(doc['abstract'])}\n")



if __name__ == "__main__":
    scrape_polynoe()
    preprocess_text()
    create_inverted_index()
    search_query = input("Enter your search query: ")
    #filters = input("Enter your filter: ")
    search(search_query)
