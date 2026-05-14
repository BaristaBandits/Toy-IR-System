from util import *

# Add your import statements here
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def __init__(self):
		# Load spaCy model (students may use this if needed)
		self.nlp = spacy.load("en_core_web_sm")

	def naive(self, text):
		#segmentedText = None

		# Fill in code here
		sentences = re.split(r'[.!?]+', text)
		segmentedText = [s.strip() for s in sentences if s.strip()]

		return segmentedText


	def punkt(self, text):

		#segmentedText = None

		# Fill in code here
		segmentedText = sent_tokenize(text)

		return segmentedText


	def spacySegmenter(self, text):

		#segmentedText = None

		# Fill in code here
		doc = self.nlp(text)
		segmentedText = [sent.text.strip() for sent in doc.sents]

		return segmentedText
