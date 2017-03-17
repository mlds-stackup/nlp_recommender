import nltk
import csv
from gensim.models import Word2Vec

# PROCESSING NOUNS...
print("processing... training ... \n")
f = open('products.csv', 'rt')
reader = csv.reader(f)
corpus = []
productList = []
products = []
for row in reader:
	lines = row[1]
	is_noun = lambda pos: pos[:2] == 'NN'
	tokenized = nltk.word_tokenize(lines)
	nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
	corpus.append(nouns)
	productList.append(row[0])
	products.append(row)
f.close()

# TRAINING ...
# print(corpus)
model = Word2Vec(corpus)

# PREDICT!
predWord = 'Chair'
preds = model.most_similar(predWord, topn=4)
print("products similar to :",predWord,'\n')
recommendations = []
for pred in preds:
	for productIndex,words in enumerate(corpus):
		for word in words:
			if word == pred[0]:
				recommendations.append(productIndex)

for reco in recommendations[:5]:
	print(products[reco][0],') ',products[reco][1])

print("\nThank you!\n")

# recommendations = []
# for searchIndex,corp in enumerate(corpus[:1]):
# 	for noun in corp:
# 		_recommendations = []
# 		preds = model.most_similar(noun, topn=4)
# 		recommendations = []
# 		for pred in preds:
# 			for productIndex,words in enumerate(corpus):
# 				for word in words:
# 					if word == pred[0]:
# 						_recommendations.append(productIndex)

# 		for reco in recommendations[:5]:
# 			recommendations.append(searchIndex,_recommendations)
