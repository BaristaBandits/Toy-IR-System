from util import *

# Add your import statements here
from nltk.corpus import stopwords



class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		#stopwordRemovedText = None

		#Fill in code here
		stop_words = set(stopwords.words('english'))
		stopwordRemovedText = []
		for sentence in text:
			filtered = [word for word in sentence if word.lower() not in stop_words]
			stopwordRemovedText.append(filtered)


		return stopwordRemovedText




	