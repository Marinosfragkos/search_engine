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

# Εισαγωγή απαραίτητων βιβλιοθηκών για εξαγωγή δεδομένων από τον ιστό, προεπεξεργασία κειμένου και αξιολόγηση αλγορίθμων αναζήτησης:
# - requests: Επιτρέπει την αποστολή HTTP αιτημάτων για την απόκτηση περιεχομένου ιστού.
# - BeautifulSoup: Αναλύει HTML περιεχόμενο για την εξαγωγή σχετικών δεδομένων.
# - json: Χρησιμοποιείται για τη διαχείριση δεδομένων σε μορφή JSON για δομημένη είσοδο και έξοδο.
# - nltk: Βιβλιοθήκη για επεξεργασία φυσικής γλώσσας, που χρησιμοποιείται για τον τεμαχισμό λέξεων, την αφαίρεση άχρηστων λέξεων, τη μετοχή και τη λεμματοποίηση.
# - sklearn: Παρέχει εργαλεία μηχανικής μάθησης, όπως η μεθοδολογία TF-IDF και ο υπολογισμός ομοιότητας cosine.
# - rank_bm25: Υλοποιεί τον αλγόριθμο BM25 για την κατάταξη εγγράφων βάσει της σχετικότητάς τους.
# - numpy: Προσφέρει αποδοτικές αριθμητικές λειτουργίες για επεξεργασία δεδομένων και υπολογισμούς σε πίνακες.

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
# Η συνάρτηση scrape_polynoe() εξάγει δεδομένα από τη σελίδα του αποθετηρίου Polynoe. 
# Συγκεκριμένα, αντλεί πληροφορίες για τον τίτλο, τον συγγραφέα, την ημερομηνία και την περίληψη κάθε εγγράφου.
# Τα δεδομένα συλλέγονται από τα στοιχεία HTML της σελίδας, οργανώνονται σε λίστα και αποθηκεύονται σε αρχείο JSON.

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
# Η συνάρτηση preprocess_text() προεπεξεργάζεται τα δεδομένα που έχουν εξαχθεί από το αρχείο JSON. 
# Αφαιρεί άχρηστες λέξεις (stop words), διατηρεί μόνο αλφαβητικούς χαρακτήρες, και εφαρμόζει διαδικασίες μετοχής (stemming) και λεμματοποίησης (lemmatization).
# Οι προεπεξεργασμένες πληροφορίες (τίτλος, συγγραφέας, περίληψη) αποθηκεύονται σε νέο αρχείο JSON για περαιτέρω ανάλυση ή χρήση.

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
# Η συνάρτηση create_inverted_index() δημιουργεί ένα αντεστραμμένο ευρετήριο (inverted index) από τα δεδομένα του αρχείου JSON που περιέχουν τις προεπεξεργασμένες πληροφορίες.
# Κάθε λέξη από την περίληψη των εγγράφων αντιστοιχεί σε ένα σύνολο από έγγραφα (δείκτες) όπου εμφανίζεται.
# Το αντίστροφο ευρετήριο αποθηκεύεται σε νέο αρχείο JSON για να χρησιμοποιηθεί σε μελλοντική αναζήτηση ή ανάλυση.

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
# Η συνάρτηση search() επιτρέπει στον χρήστη να επιλέξει έναν αλγόριθμο αναζήτησης για την εκτέλεση της αναζήτησης με βάση το δεδομένο ερώτημα.
# Παρουσιάζει τρεις επιλογές: Boolean Retrieval, Vector Space Model και Okapi BM25.
# Ανάλογα με την επιλογή του χρήστη, καλεί την αντίστοιχη συνάρτηση αναζήτησης και εμφανίζει τα αποτελέσματα.
# Αν η επιλογή είναι μη έγκυρη, εμφανίζει μήνυμα λάθους.

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


# Η συνάρτηση boolean_retrieval() εκτελεί αναζήτηση βάσει του μοντέλου Boolean, χρησιμοποιώντας τον αντεστραμμένο ευρετήριο.
# Αρχικά, επεξεργάζεται το ερώτημα και φορτώνει τον αντίστροφο ευρετήριο από το αρχείο JSON.
# Στη συνέχεια, εφαρμόζει τους λογικούς τελεστές 'AND', 'OR' και 'NOT' για να εντοπίσει τα σχετικά έγγραφα.
# Η συνάρτηση επιστρέφει τη λίστα των εγγράφων που πληρούν τις συνθήκες του ερωτήματος.

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

# Η συνάρτηση vector_space_model() εκτελεί αναζήτηση χρησιμοποιώντας το μοντέλο του διανύσματος χώρου (Vector Space Model) με βάση την ομοιότητα cosine.
# Αρχικά, φορτώνει τα προεπεξεργασμένα έγγραφα και το ερώτημα, το οποίο αναλύεται σε λέξεις (tokenization).
# Υπολογίζει τη συχνότητα εμφάνισης λέξεων (TF) και την αντίστοιχη συχνότητα αντεστραμμένης εμφάνισης (IDF) για όλα τα έγγραφα και το ερώτημα.
# Στη συνέχεια, υπολογίζει την ομοιότητα cosine ανάμεσα στο ερώτημα και τα έγγραφα, και τα ταξινομεί κατά σειρά ομοιότητας.
# Η συνάρτηση επιστρέφει τα έγγραφα ταξινομημένα με βάση την ομοιότητά τους με το ερώτημα.

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

# Η συνάρτηση okapibm25() εκτελεί αναζήτηση χρησιμοποιώντας τον αλγόριθμο Okapi BM25, ο οποίος αξιολογεί τη συσχέτιση των εγγράφων με το ερώτημα.
# Αρχικά, φορτώνει τα προεπεξεργασμένα έγγραφα και το ερώτημα, το οποίο διασπάται σε λέξεις (tokenization).
# Στη συνέχεια, ο αλγόριθμος BM25 υπολογίζει τις βαθμολογίες ομοιότητας για κάθε έγγραφο και επιστρέφει τα καλύτερα αποτελέσματα ταξινομημένα κατά τις βαθμολογίες τους.

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


# Φιλτράρισμα Αποτελεσμάτων 
# def filter_results(criteria, value):
#     # Άνοιγμα του αρχείου με τα επεξεργασμένα δεδομένα
#     with open('processed_data.json', 'r', encoding='utf8') as f:
#         data = json.load(f)

#     # Δημιουργία μιας λίστας με τα έγγραφα που πληρούν το κριτήριο
#     filtered_data = [doc for doc in data if doc.get(criteria) == value]
#         print(filtered_data)

#     # Επιστροφή της λίστας με τα φιλτραρισμένα δεδομένα
#     return filtered_data

# Επεξεργασία ερωτήματος (Query Processing)
# Η συνάρτηση query_processing() προεπεξεργάζεται το ερώτημα αναζήτησης με σκοπό να το καταστήσει πιο αποδοτικό για την αναζήτηση.
# Αρχικά, διαγράφονται οι άχρηστες λέξεις (stop words), έπειτα οι λέξεις του ερωτήματος μετατρέπονται σε μικρά γράμματα.
# Στη συνέχεια, εφαρμόζεται το stemming και η λεμματοποίηση για να μειωθούν οι λέξεις στην βασική τους μορφή.
# Τέλος, επιστρέφει τη λίστα με τις επεξεργασμένες λέξεις του ερωτήματος.

def query_processing(query):

    stop_words = set(stopwords.words('greek'))  # Define stop_words
    stemmer = PorterStemmer()  # Define stemmer
    lemmatizer = WordNetLemmatizer()  # Define lemmatizer

    query_tokens = word_tokenize(query)
    query_tokens = [word.lower() for word in query_tokens if word.isalpha() and word.lower() not in stop_words]
    query_tokens = [stemmer.stem(word) for word in query_tokens]
    query_tokens = [lemmatizer.lemmatize(word) for word in query_tokens]
    
    return query_tokens

# Κατάταξη αποτελεσμάτων (Ranking)
# Η συνάρτηση ranking() επιλέγει και εφαρμόζει έναν αλγόριθμο ταξινόμησης για να επιστρέψει τα πιο σχετικά αποτελέσματα για το δοθέν ερώτημα.
# Ανάλογα με την τιμή του παραμέτρου `ranking_algorithm`, καλείται είτε ο αλγόριθμος του Vector Space Model είτε ο αλγόριθμος Okapi BM25.
# Στη συνέχεια, εκτυπώνει τα αποτελέσματα ταξινομημένα κατά τη σχετικότητα τους, εμφανίζοντας τις βασικές πληροφορίες (τίτλος, συγγραφέας, ημερομηνία, περίληψη).

def ranking(query, ranking_algorithm):
    if ranking_algorithm == 'vectorspacemodel':
        results = vector_space_model(query)
    elif ranking_algorithm == 'okapibm25':
        results = okapibm25(query)
    else:
        raise ValueError("Unsupported ranking algorithm")
    for doc, similarity in results:
        print(f"Similarity: {similarity:.2f}\nTitle: {' '.join(doc['title'])}\nAuthor: {' '.join(doc['author'])}\nDate: {doc['date']}\nAbstract: {' '.join(doc['abstract'])}\n")

# Αυτή η συνάρτηση είναι η κύρια συνάρτηση του προγράμματος. 
# Ξεκινά με την εξαγωγή των δεδομένων από τον ιστότοπο Polynoe (scrape_polynoe), 
# την προεπεξεργασία των δεδομένων (preprocess_text), και τη δημιουργία του αντίστροφου ευρετηρίου (create_inverted_index). 
# Στη συνέχεια, ζητά από τον χρήστη να εισάγει ένα ερώτημα αναζήτησης και καλεί τη συνάρτηση αναζήτησης (search) για να εμφανίσει τα αποτελέσματα.

if __name__ == "__main__":
    scrape_polynoe()
    preprocess_text()
    create_inverted_index()
    search_query = input("Enter your search query: ")
    #filters = input("Enter your filter: ")
    search(search_query)
