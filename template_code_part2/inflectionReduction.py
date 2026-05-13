from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class InflectionReduction:

	def porterStemmer(self, text):
		"""
		Inflection Reduction using Porter Stemmer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed tokens representing a sentence
		"""

		#reducedText = None

		# Fill in code here
		stemmer = PorterStemmer()
		reducedText = []
		for sentence in text:
			stems = [stemmer.stem(word) for word in sentence]
			reducedText.append(stems)

		return reducedText



	def wordnetLemmatizer(self, text):
		"""
		Inflection Reduction using WordNet Lemmatizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			lemmatized tokens representing a sentence
		"""

		#reducedText = None

		# Fill in code here
		lemmatizer = WordNetLemmatizer()
		reducedText = []
		for sentence in text:
			lemmas = [lemmatizer.lemmatize(word) for word in sentence]
			reducedText.append(lemmas)

		return reducedText



	def reduce(self, text):
		"""
		Wrapper function for inflection reduction.
		Students may choose which method to call
		or extend this function to support both options.
		"""

		#reducedText = None

		# Fill in code here
		reducedText = self.wordnetLemmatizer(text)

		return reducedText
