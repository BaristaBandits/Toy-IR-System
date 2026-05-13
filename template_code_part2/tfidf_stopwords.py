import json
import os
import argparse
from collections import Counter

def get_data_driven_stopwords(dataset_path, threshold=0.3):
    """
    Identifies stopwords based on Document Frequency.
    threshold=0.3 means any word appearing in more than 30% of docs is a stopword.
    """
    doc_path = os.path.join(dataset_path, "cran_docs.json")
    
    with open(doc_path, 'r') as f:
        docs = json.load(f)
    
    total_docs = len(docs)
    word_doc_counts = Counter()

    for item in docs:
        # Get unique words in this specific document
        words = set(item["body"].lower().split())
        for word in words:
            # Clean punctuation from words
            clean_word = "".join(char for char in word if char.isalnum())
            if clean_word:
                word_doc_counts[clean_word] += 1

    # Identify words that appear in more than 'threshold' percent of documents
    custom_stopwords = [word for word, count in word_doc_counts.items() 
                        if (count / total_docs) >= threshold]
    
    return sorted(custom_stopwords)

if __name__ == "__main__":
    # Update this path to your actual cranfield folder path
    dataset = "C:/#Mana_files/IITM academics/Sem 8/cs6370/NLP Assignment 1/Cranfield Dataset NLP/cranfield"
    
    print(f"--- Analyzing corpus at {dataset} ---")
    stop_list = get_data_driven_stopwords(dataset)
    
    print(f"\nFound {len(stop_list)} data-driven stopwords:")
    print(stop_list)
    
    # Save to a file for your assignment
    with open("data_driven_stopwords.txt", "w") as f:
        f.write("\n".join(stop_list))
    print("\nResults saved to 'data_driven_stopwords.txt'")