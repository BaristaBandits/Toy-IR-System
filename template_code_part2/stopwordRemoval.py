from util import *

# Add your import statements here
from nltk.corpus import stopwords



class StopwordRemoval():

	def fromList(self, text):
	
		#stopwordRemovedText = None

		#Fill in code here
		stop_words = set(stopwords.words('english'))
		stopwordRemovedText = []
		for sentence in text:
			filtered = [word for word in sentence if word.lower() not in stop_words]
			stopwordRemovedText.append(filtered)


		return stopwordRemovedText




	
