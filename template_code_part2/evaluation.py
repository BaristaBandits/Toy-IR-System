from util import *

# Add your import statements here
import math


class Evaluation():

	def getRelevantDocs(self, query_id, qrels):

		relevant_docs = []

		for item in qrels:

			qid = int(item["query_num"])
			docid = int(item["id"])

			if qid == int(query_id):
				relevant_docs.append(docid)

		return relevant_docs
	


	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		retrieved_docs = query_doc_IDs_ordered[:k]

		relevant_count = 0

		for docID in retrieved_docs:
			if docID in true_doc_IDs:
				relevant_count += 1

		precision = relevant_count / k if k > 0 else 0

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):

		precisions = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, qrels)
			precision = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(precision)

		meanPrecision = sum(precisions) / len(precisions)

		return meanPrecision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		retrieved_docs = query_doc_IDs_ordered[:k]

		relevant_count = 0

		for docID in retrieved_docs:
			if docID in true_doc_IDs:
				relevant_count += 1

		recall = (
			relevant_count / len(true_doc_IDs)
			if len(true_doc_IDs) > 0
			else 0
		)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):

		recalls = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, qrels)
			recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(recall)

		meanRecall = sum(recalls) / len(recalls)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		beta = 0.5

		if precision == 0 and recall == 0:
			return 0

		fscore = (
			(1 + beta**2) * precision * recall
		) / (
			(beta**2 * precision) + recall
		)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):

		fscores = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, qrels)
			fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(fscore)

		meanFscore = sum(fscores) / len(fscores)

		return meanFscore


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		retrieved_docs = query_doc_IDs_ordered[:k]

		# DCG
		DCG = 0

		for i, docID in enumerate(retrieved_docs):

			if docID in true_doc_IDs:
				relevance = 1
			else:
				relevance = 0

			DCG += relevance / math.log2(i + 2)

		# IDCG
		ideal_relevance_count = min(len(true_doc_IDs), k)

		IDCG = 0

		for i in range(ideal_relevance_count):
			IDCG += 1 / math.log2(i + 2)

		if IDCG == 0:
			return 0

		nDCG = DCG / IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):

		nDCGs = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, qrels)
			nDCG = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			nDCGs.append(nDCG)

		meanNDCG = sum(nDCGs) / len(nDCGs)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		retrieved_docs = query_doc_IDs_ordered[:k]

		relevant_count = 0

		precision_sum = 0

		for i, docID in enumerate(retrieved_docs):

			if docID in true_doc_IDs:

				relevant_count += 1

				precision_at_i = relevant_count / (i + 1)

				precision_sum += precision_at_i

		if relevant_count == 0:
			return 0

		avgPrecision = precision_sum / relevant_count

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):

		APs = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, q_rels)

			AP = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs,k)
			APs.append(AP)

		meanAveragePrecision = sum(APs) / len(APs)

		return meanAveragePrecision


	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):

		retrieved_docs = query_doc_IDs_ordered[:k]

		for i, docID in enumerate(retrieved_docs):
			if docID in true_doc_IDs:
				return 1 / (i + 1)

		return 0


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):

		RRs = []

		for i, query_id in enumerate(query_ids):

			true_doc_IDs = self.getRelevantDocs(query_id, qrels)
			RR = self.queryReciprocalRank(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			RRs.append(RR)

		meanReciprocalRank = sum(RRs) / len(RRs)

		return meanReciprocalRank
