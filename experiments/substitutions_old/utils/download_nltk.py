import nltk
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    print("WordNet already downloaded.")
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
