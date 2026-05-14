from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, WordNetLemmatizer, PorterStemmer, etc.)
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class InflectionReduction:

	def porterStemmer(self, text):
		#reducedText = None

		# Fill in code here
		stemmer = PorterStemmer()
		reducedText = []
		for sentence in text:
			stems = [stemmer.stem(word) for word in sentence]
			reducedText.append(stems)

		return reducedText



	def wordnetLemmatizer(self, text):

		#reducedText = None

		# Fill in code here
		lemmatizer = WordNetLemmatizer()
		reducedText = []
		for sentence in text:
			lemmas = [lemmatizer.lemmatize(word) for word in sentence]
			reducedText.append(lemmas)

		return reducedText



	def reduce(self, text):
		#reducedText = None

		# Fill in code here
		reducedText = self.wordnetLemmatizer(text)

		return reducedText
