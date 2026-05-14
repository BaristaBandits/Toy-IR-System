from util import *

# Add your import statements here
# (Students may import required libraries such as nltk, spacy, re, etc.)
import re
from nltk.tokenize import TreebankWordTokenizer
import spacy

class Tokenization():

	def naive(self, text):


		#tokenizedText = None

		# Fill in code here
		tokenizedText = []
		for sentence in text:
			tokens = re.findall(r'\b\w+\b', sentence.lower())
			tokenizedText.append(tokens)


		return tokenizedText



	def pennTreeBank(self, text):
		

		#tokenizedText = None

		# Fill in code here
		tokenizer = TreebankWordTokenizer()
		tokenizedText = []
		for sentence in text:
			tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText



	def spacyTokenizer(self, text):
		

		#tokenizedText = None

		# Fill in code here
		nlp = spacy.load("en_core_web_sm")
		tokenizedText = []
		for sentence in text:
			doc = nlp(sentence)
			tokens = [token.text for token in doc]
			tokenizedText.append(tokens)


		return tokenizedText
