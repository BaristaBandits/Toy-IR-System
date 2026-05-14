from util import *

import math
import numpy as np

from collections import Counter

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from nltk.corpus import wordnet
from difflib import get_close_matches


class ImprovedInformationRetrieval():

    def __init__(
        self,
        method="baseline",
        n_components=100
    ):

        self.method = method
        self.n_components = n_components

        self.docIDs = None
        self.vocab = None
        self.idf = None
        self.doc_vectors = None
        self.lsa_pipeline = None

        self.flattened_docs = None
        
    # QUERY EXPANSION
    def expand_query(self, query_tokens):

        expanded = list(query_tokens)

        for word in query_tokens:
            synsets = wordnet.synsets(word)
            for syn in synsets[:2]:
                for lemma in syn.lemmas()[:2]:
                    synonym = lemma.name().lower()
                    synonym = synonym.replace("_", " ")
                    if synonym not in expanded:
                        expanded.append(synonym)

        return expanded


    # SPELL CORRECTION
    def correct_query(self, query_tokens):

        corrected = []

        for word in query_tokens:
            if word in self.vocab:
                corrected.append(word)

            else:
                matches = get_close_matches( word,self.vocab,n=1,cutoff=0.8)
                if matches:
                    corrected.append(matches[0])
                else:
                    corrected.append(word)

        return corrected

    # BIGRAMS
    def add_bigrams(self, tokens):
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            bigrams.append(bigram)

        return tokens + bigrams
        
    # BUILD INDEX
    def buildIndex(self, docs, docIDs):

        self.docIDs = docIDs
        flattened_docs = []
        for doc in docs:
            tokens = []
            for sentence in doc:
                tokens.extend(sentence)
            # BIGRAM MODELS
            if self.method in ["bigram", "hybrid_bigram" ]:
                tokens = self.add_bigrams(tokens)
            flattened_docs.append(tokens)
        self.flattened_docs = flattened_docs

        # VOCAB
        vocab = set()

        for doc in flattened_docs:
            vocab.update(doc)

        self.vocab = sorted(list(vocab))

        # DOCUMENT FREQUENCY
        df = {}

        for term in self.vocab:
            df[term] = 0
            for doc in flattened_docs:
                if term in doc:
                    df[term] += 1

        # IDF
        N = len(flattened_docs)
        self.idf = {}
        for term in self.vocab:
            
            # SMART TF-IDF
            if self.method == "smart":
                self.idf[term] =  math.log((N + 1) / (df[term] + 1)) + 1

            else:
                self.idf[term] = math.log(N / (df[term] + 1)
        
        # TF-IDF DOCUMENT MATRIX
        doc_matrix = []
        for doc in flattened_docs:
            tf = Counter(doc)
            vector = np.zeros(len(self.vocab))
            total_terms = len(doc)
            for j, term in enumerate(self.vocab):

                #SMART
                if self.method == "smart":

                    if tf[term] > 0:
                        tf_value = 1 + math.log(tf[term])
                        
                    else:
                        tf_value = 0

                else:

                    tf_value = tf[term] / total_terms if total_terms > 0 else 0

                vector[j] = tf_value * self.idf[term]

            doc_matrix.append(vector)
        doc_matrix = np.array(doc_matrix)

        
        # LSA + hybrid methods
        if self.method in ["lsa", "hybrid", "hybrid_bigram"]:
        
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        
            normalizer = Normalizer(copy=False)
        
            self.lsa_pipeline = make_pipeline(svd, normalizer)
        
            self.doc_vectors = self.lsa_pipeline.fit_transform(doc_matrix)
        
        else:
            self.doc_vectors = doc_matrix
        
        
        def rank(self, queries):
        
            doc_IDs_ordered = []
        
            for query in queries:
        
                # flatten nested query
                query_tokens = []
        
                for sentence in query:
                    query_tokens.extend(sentence)
        
                # fix spelling mistakes
                if self.method in ["spell", "bm25_spell"]:
                    query_tokens = self.correct_query(query_tokens)
        
                # add similar terms
                if self.method in ["qe", "hybrid", "hybrid_bigram"]:
                    query_tokens = self.expand_query(query_tokens)
        
                # add bigrams
                if self.method in ["bigram", "hybrid_bigram"]:
                    query_tokens = self.add_bigrams(query_tokens)
        
                # create tf-idf query vector
                tf_query = Counter(query_tokens)
        
                query_vector = np.zeros(len(self.vocab))
        
                total_terms = len(query_tokens)
        
                for i, term in enumerate(self.vocab):
        
                    if total_terms > 0:
                        tf_value = tf_query[term] / total_terms
                    else:
                        tf_value = 0
        
                    if self.method == "boosted":
        
                        # boost rare words more
                        query_vector[i] = tf_value * (self.idf.get(term, 0) ** 2)
        
                    else:
        
                        query_vector[i] = tf_value * self.idf.get(term, 0)
        
                # project into latent semantic space
                if self.method in ["lsa", "hybrid", "hybrid_bigram"]:
                    query_vector = self.lsa_pipeline.transform([query_vector])[0]
        
                # cosine similarity with documents
                scores = []
        
                for i, docID in enumerate(self.docIDs):
        
                    doc_vector = self.doc_vectors[i]
        
                    numerator = np.dot(query_vector, doc_vector)
        
                    denominator = (
                        np.linalg.norm(query_vector)
                        * np.linalg.norm(doc_vector)
                    )
        
                    if denominator == 0:
                        similarity = 0
                    else:
                        similarity = numerator / denominator
        
                    scores.append((docID, similarity))
        
                # sort by score
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
                ranked_docIDs = []
        
                for docID, score in scores:
                    ranked_docIDs.append(docID)
        
                doc_IDs_ordered.append(ranked_docIDs)
        
            return doc_IDs_ordered
