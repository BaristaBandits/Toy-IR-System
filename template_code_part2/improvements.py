from util import *

import math
import numpy as np

from collections import Counter

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from nltk.corpus import wordnet


class ImprovedInformationRetrieval():

    def __init__(self,
                 method="baseline",
                 n_components=100):

        self.method = method
        self.n_components = n_components

        self.docIDs = None
        self.vocab = None
        self.idf = None

        self.doc_vectors = None

        self.lsa_pipeline = None


    ########################################################
    # QUERY EXPANSION
    ########################################################

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


    ########################################################
    # BUILD INDEX
    ########################################################

    def buildIndex(self, docs, docIDs):

        self.docIDs = docIDs

        flattened_docs = []

        for doc in docs:

            tokens = []

            for sentence in doc:
                tokens.extend(sentence)

            flattened_docs.append(tokens)

        ####################################################
        # VOCAB
        ####################################################

        vocab = set()

        for doc in flattened_docs:
            vocab.update(doc)

        self.vocab = sorted(list(vocab))

        ####################################################
        # DF
        ####################################################

        df = {}

        for term in self.vocab:

            df[term] = 0

            for doc in flattened_docs:

                if term in doc:
                    df[term] += 1

        ####################################################
        # IDF
        ####################################################

        N = len(flattened_docs)

        self.idf = {}

        for term in self.vocab:

            self.idf[term] = math.log(
                N / (df[term] + 1)
            )

        ####################################################
        # TF-IDF DOCUMENT MATRIX
        ####################################################

        doc_matrix = []

        for doc in flattened_docs:

            tf = Counter(doc)

            vector = np.zeros(len(self.vocab))

            total_terms = len(doc)

            for j, term in enumerate(self.vocab):

                tf_value = (
                    tf[term] / total_terms
                    if total_terms > 0
                    else 0
                )

                vector[j] = tf_value * self.idf[term]

            doc_matrix.append(vector)

        doc_matrix = np.array(doc_matrix)

        ####################################################
        # BASELINE
        ####################################################

        if self.method in ["baseline", "qe"]:

            self.doc_vectors = doc_matrix

        ####################################################
        # LSA / HYBRID
        ####################################################

        elif self.method in ["lsa", "hybrid"]:

            svd = TruncatedSVD(
                n_components=self.n_components,
                random_state=42
            )

            normalizer = Normalizer(copy=False)

            self.lsa_pipeline = make_pipeline(
                svd,
                normalizer
            )

            self.doc_vectors = self.lsa_pipeline.fit_transform(
                doc_matrix
            )


    ########################################################
    # RANK
    ########################################################

    def rank(self, queries):

        doc_IDs_ordered = []

        for query in queries:

            ################################################
            # FLATTEN QUERY
            ################################################

            query_tokens = []

            for sentence in query:
                query_tokens.extend(sentence)

            ################################################
            # QUERY EXPANSION
            ################################################

            if self.method in ["qe", "hybrid"]:

                query_tokens = self.expand_query(
                    query_tokens
                )

            ################################################
            # QUERY TF-IDF VECTOR
            ################################################

            tf_query = Counter(query_tokens)

            query_vector = np.zeros(len(self.vocab))

            total_terms = len(query_tokens)

            for i, term in enumerate(self.vocab):

                tf_value = (
                    tf_query[term] / total_terms
                    if total_terms > 0
                    else 0
                )

                query_vector[i] = (
                    tf_value *
                    self.idf.get(term, 0)
                )

            ################################################
            # LSA PROJECTION
            ################################################

            if self.method in ["lsa", "hybrid"]:

                query_vector = self.lsa_pipeline.transform(
                    [query_vector]
                )[0]

            ################################################
            # COSINE SIMILARITY
            ################################################

            scores = []

            for i, docID in enumerate(self.docIDs):

                doc_vector = self.doc_vectors[i]

                numerator = np.dot(
                    query_vector,
                    doc_vector
                )

                denominator = (
                    np.linalg.norm(query_vector)
                    * np.linalg.norm(doc_vector)
                )

                if denominator == 0:
                    similarity = 0
                else:
                    similarity = numerator / denominator

                scores.append(
                    (docID, similarity)
                )

            ################################################
            # SORT
            ################################################

            scores = sorted(
                scores,
                key=lambda x: x[1],
                reverse=True
            )

            ranked_docIDs = [
                docID for docID, score in scores
            ]

            doc_IDs_ordered.append(
                ranked_docIDs
            )

        return doc_IDs_ordered

