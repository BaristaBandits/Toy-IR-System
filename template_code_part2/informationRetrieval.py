from util import *

# Add your import statements here
import math
import numpy as np
from collections import Counter


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None
		self.vocab = None
		self.idf = None
		self.doc_vectors = None


	def buildIndex(self, docs, docIDs):
		self.docIDs = docIDs

		# Flatten documents
		flattened_docs = []

		for doc in docs:
			tokens = []

			for sentence in doc:
				tokens.extend(sentence)

			flattened_docs.append(tokens)

		# Build vocabulary
		vocab = set()

		for doc in flattened_docs:
			for word in doc:
				vocab.add(word)

		self.vocab = sorted(list(vocab))

		# Compute Document Frequency (DF)
		df = {}

		for term in self.vocab:
			df[term] = 0

			for doc in flattened_docs:
				if term in doc:
					df[term] += 1

		# Compute IDF
		N = len(flattened_docs)

		self.idf = {}

		for term in self.vocab:
			self.idf[term] = math.log(N / (df[term] + 1))

		# Compute TF-IDF vectors
		self.doc_vectors = {}

		for i, doc in enumerate(flattened_docs):
			tf = Counter(doc)
			vector = np.zeros(len(self.vocab))
			total_terms = len(doc)
			for j, term in enumerate(self.vocab):
				tf_value = tf[term] / total_terms if total_terms > 0 else 0
				vector[j] = tf_value * self.idf[term]
			self.doc_vectors[docIDs[i]] = vector
		self.index = self.doc_vectors


	def rank(self, queries):
	

		doc_IDs_ordered = []

		for query in queries:
			query_tokens = [] #flatten query
			for sentence in query:
				query_tokens.extend(sentence)
			# Compute query TF
			tf_query = Counter(query_tokens)
			query_vector = np.zeros(len(self.vocab))
			total_terms = len(query_tokens)
			for i, term in enumerate(self.vocab):
				tf_value = tf_query[term] / total_terms if total_terms > 0 else 0
				query_vector[i] = tf_value * self.idf.get(term, 0)

			# Compute cosine similarities
			scores = []
			for docID in self.docIDs:
				doc_vector = self.doc_vectors[docID]
				numerator = np.dot(query_vector, doc_vector)
				denominator = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
				if denominator == 0:
					similarity = 0
				else:
					similarity = numerator / denominator

				scores.append((docID, similarity))

			# Sort by similarity descending
			scores = sorted(scores, key=lambda x: x[1], reverse=True)
			ranked_docIDs = [docID for docID, score in scores]
			doc_IDs_ordered.append(ranked_docIDs)

		return doc_IDs_ordered
