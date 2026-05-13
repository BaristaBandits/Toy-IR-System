from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval

# IMPORT BOTH
from informationRetrieval import InformationRetrieval
from improvements import ImprovedInformationRetrieval

from evaluation import Evaluation

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
import os
import time

# Input compatibility
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass


class SearchEngine:

    def __init__(self, args):

        self.args = args

        if not os.path.exists(self.args.out_folder):
            os.makedirs(self.args.out_folder)

        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

        self.evaluator = Evaluation()

    # ---------------------------------------------------------
    # PREPROCESSING
    # ---------------------------------------------------------

    def segmentSentences(self, text):

        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)

        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):

        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)

        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):

        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):

        return self.stopwordRemover.fromList(text)

    # ---------------------------------------------------------
    # QUERY PREPROCESSING
    # ---------------------------------------------------------

    def preprocessQueries(self, queries):

        segmentedQueries = []

        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)

        tokenizedQueries = []

        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)

        reducedQueries = []

        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)

        stopwordRemovedQueries = []

        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)

        return stopwordRemovedQueries

    # ---------------------------------------------------------
    # DOC PREPROCESSING
    # ---------------------------------------------------------

    def preprocessDocs(self, docs):

        segmentedDocs = []

        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)

        tokenizedDocs = []

        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)

        reducedDocs = []

        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)

        stopwordRemovedDocs = []

        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)

        return stopwordRemovedDocs

    # ---------------------------------------------------------
    # RUN ONE METHOD
    # ---------------------------------------------------------

    def run_method(
        self,
        method_name,
        retriever,
        processedDocs,
        processedQueries,
        doc_ids,
        query_ids,
        qrels
    ):

        print("\n======================================")
        print("RUNNING:", method_name)
        print("======================================")

        start_time = time.time()

        retriever.buildIndex(processedDocs, doc_ids)

        doc_IDs_ordered = retriever.rank(processedQueries)

        precisions = []
        recalls = []
        fscores = []
        MAPs = []
        nDCGs = []
        MRRs = []

        for k in range(1, 11):

            precision = self.evaluator.meanPrecision(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            recall = self.evaluator.meanRecall(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            fscore = self.evaluator.meanFscore(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            MAP = self.evaluator.meanAveragePrecision(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            nDCG = self.evaluator.meanNDCG(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            MRR = self.evaluator.meanReciprocalRank(
                doc_IDs_ordered,
                query_ids,
                qrels,
                k
            )

            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)
            MAPs.append(MAP)
            nDCGs.append(nDCG)
            MRRs.append(MRR)

        end_time = time.time()

        # FINAL RESULTS
        print("\nFINAL RESULTS @10")
        print("-----------------------------")

        print(f"Precision@10 : {precisions[-1]:.4f}")
        print(f"Recall@10    : {recalls[-1]:.4f}")
        print(f"F-score@10   : {fscores[-1]:.4f}")
        print(f"MAP@10       : {MAPs[-1]:.4f}")
        print(f"nDCG@10      : {nDCGs[-1]:.4f}")
        print(f"MRR@10       : {MRRs[-1]:.4f}")

        print(f"\nRuntime: {end_time - start_time:.2f} sec")

        # SAVE PLOT
        plt.figure(figsize=(10, 6))

        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.plot(range(1, 11), MRRs, label="MRR")

        plt.xlabel("k")
        plt.ylabel("Score")
        plt.title(f"Metrics - {method_name}")

        plt.legend()

        plt.savefig(
            os.path.join(
                self.args.out_folder,
                f"{method_name}_plot.png"
            )
        )

        plt.close()

    # ---------------------------------------------------------
    # EVALUATE ALL
    # ---------------------------------------------------------

    def evaluateDataset(self):

        # LOAD QUERIES
        queries_json = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_queries.json"
                ),
                'r'
            )
        )[:]

        query_ids = [
            item["query number"]
            for item in queries_json
        ]

        queries = [
            item["query"]
            for item in queries_json
        ]

        processedQueries = self.preprocessQueries(queries)

        # LOAD DOCS
        docs_json = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_docs.json"
                ),
                'r'
            )
        )[:]

        doc_ids = [
            item["id"]
            for item in docs_json
        ]

        docs = [
            item["body"]
            for item in docs_json
        ]

        processedDocs = self.preprocessDocs(docs)

        # LOAD QRELS
        qrels = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_qrels.json"
                ),
                'r'
            )
        )[:]

        # -------------------------------------------------
        # BASELINE
        # -------------------------------------------------

        baseline_ir = InformationRetrieval()

        self.run_method(
            "baseline",
            baseline_ir,
            processedDocs,
            processedQueries,
            doc_ids,
            query_ids,
            qrels
        )

        # -------------------------------------------------
        # LSA
        # -------------------------------------------------

        lsa_ir = ImprovedInformationRetrieval(
            method="lsa",
            n_components=100
        )

        self.run_method(
            "lsa",
            lsa_ir,
            processedDocs,
            processedQueries,
            doc_ids,
            query_ids,
            qrels
        )

        # -------------------------------------------------
        # QUERY EXPANSION
        # -------------------------------------------------

        qe_ir = ImprovedInformationRetrieval(
            method="qe"
        )

        self.run_method(
            "query_expansion",
            qe_ir,
            processedDocs,
            processedQueries,
            doc_ids,
            query_ids,
            qrels
        )

        # -------------------------------------------------
        # HYBRID
        # -------------------------------------------------

        hybrid_ir = ImprovedInformationRetrieval(
            method="hybrid",
            n_components=100
        )

        self.run_method(
            "hybrid",
            hybrid_ir,
            processedDocs,
            processedQueries,
            doc_ids,
            query_ids,
            qrels
        )

    # ---------------------------------------------------------
    # CUSTOM QUERY
    # ---------------------------------------------------------

    def handleCustomQuery(self):

        print("Enter query below")

        query = input()

        processedQuery = self.preprocessQueries([query])[0]

        docs_json = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_docs.json"
                ),
                'r'
            )
        )[:]

        doc_ids = [
            item["id"]
            for item in docs_json
        ]

        docs = [
            item["body"]
            for item in docs_json
        ]

        processedDocs = self.preprocessDocs(docs)

        retriever = InformationRetrieval()

        retriever.buildIndex(processedDocs, doc_ids)

        doc_IDs_ordered = retriever.rank(
            [processedQuery]
        )[0]

        print("\nTop five document IDs:")

        for id_ in doc_IDs_ordered[:5]:
            print(id_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='main.py'
    )

    parser.add_argument(
        '-dataset',
        default="cranfield/"
    )

    parser.add_argument(
        '-out_folder',
        default="output/"
    )

    parser.add_argument(
        '-segmenter',
        default="punkt"
    )

    parser.add_argument(
        '-tokenizer',
        default="ptb"
    )

    parser.add_argument(
        '-custom',
        action="store_true"
    )

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()

    else:
        searchEngine.evaluateDataset()