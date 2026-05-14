from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from improvements import ImprovedInformationRetrieval
from evaluation import Evaluation
from sys import version_info

import argparse
import json
import matplotlib.pyplot as plt
import os
import time


# python2 compatibility
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


    # sentence splitting
    def segmentSentences(self, text):

        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)

        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)


    # tokenization
    def tokenize(self, text):

        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)

        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)


    # stemming / lemmatization
    def reduceInflection(self, text):

        return self.inflectionReducer.reduce(text)


    # remove stopwords
    def removeStopwords(self, text):

        return self.stopwordRemover.fromList(text)


    # preprocess queries
    def preprocessQueries(self, queries):

        segmentedQueries = []

        for query in queries:
            segmentedQueries.append(self.segmentSentences(query))
        tokenizedQueries = []

        for query in segmentedQueries:
            tokenizedQueries.append(self.tokenize(query))

        reducedQueries = []

        for query in tokenizedQueries:
            reducedQueries.append(self.reduceInflection(query))

        stopwordRemovedQueries = []

        for query in reducedQueries:
            stopwordRemovedQueries.append(self.removeStopwords(query) )
        return stopwordRemovedQueries


    # preprocess documents
    def preprocessDocs(self, docs):

        segmentedDocs = []
        for doc in docs:
            segmentedDocs.append(self.segmentSentences(doc) )
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDocs.append(self.tokenize(doc))
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDocs.append(self.reduceInflection(doc) )
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDocs.append( self.removeStopwords(doc))
        return stopwordRemovedDocs


    # run one retrieval model
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

        retriever.buildIndex(processedDocs,doc_ids)

        doc_IDs_ordered = retriever.rank( processedQueries)
        precisions = []
        recalls = []
        fscores = []
        MAPs = []
        nDCGs = []
        MRRs = []

        for k in range(1, 11):

            precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids,qrels,k)

            recall = self.evaluator.meanRecall(doc_IDs_ordered,query_ids,qrels,k)

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

        # print final metrics
        print("\nFINAL RESULTS @10")
        print("-----------------------------")

        print(f"Precision@10 : {precisions[-1]:.4f}")
        print(f"Recall@10    : {recalls[-1]:.4f}")
        print(f"F-score@10   : {fscores[-1]:.4f}")
        print(f"MAP@10       : {MAPs[-1]:.4f}")
        print(f"nDCG@10      : {nDCGs[-1]:.4f}")
        print(f"MRR@10       : {MRRs[-1]:.4f}")

        print(
            f"\nRuntime: {end_time - start_time:.2f} sec"
        )

        # save evaluation plot
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


    # evaluate all retrieval models
    def evaluateDataset(self):

        # load queries
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

        # load documents
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

        # load qrels
        qrels = json.load(
            open(
                os.path.join(
                    args.dataset,
                    "cran_qrels.json"
                ),
                'r'
            )
        )[:]

        # retrieval methods
        methods = [

            ("baseline", InformationRetrieval()),

            (
                "lsa",
                ImprovedInformationRetrieval(
                    method="lsa",
                    n_components=100
                )
            ),

            (
                "query_expansion",
                ImprovedInformationRetrieval(
                    method="qe"
                )
            ),

            (
                "hybrid",
                ImprovedInformationRetrieval(
                    method="hybrid",
                    n_components=100
                )
            ),

            (
                "smart_tfidf",
                ImprovedInformationRetrieval(
                    method="smart"
                )
            ),

            (
                "boosted_query",
                ImprovedInformationRetrieval(
                    method="boosted"
                )
            ),

            (
                "bigram",
                ImprovedInformationRetrieval(
                    method="bigram"
                )
            ),

            (
                "spell_correction",
                ImprovedInformationRetrieval(
                    method="spell"
                )
            ),

            (
                "hybrid_bigram",
                ImprovedInformationRetrieval(
                    method="hybrid_bigram",
                    n_components=100
                )
            )

        ]

        # run all methods
        for method_name, retriever in methods:

            self.run_method(
                method_name,
                retriever,
                processedDocs,
                processedQueries,
                doc_ids,
                query_ids,
                qrels
            )


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

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    searchEngine.evaluateDataset()
