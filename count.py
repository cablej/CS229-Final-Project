# Helper script to get most common words from a file... not used

from collections import Counter
import nltk
from nltk.corpus import stopwords

exclude = stopwords.words('english')

#opens the file. the with statement here will automatically close it afterwards.
with open("IRAhandle_tweets_1.csv") as input_file:
    #build a counter from each word in the file
    count = Counter(word for line in input_file
                         for word in line.split(",") if word.lower() not in exclude and len(word) > 2)

common = count.most_common(1000)

for word in common:
	print(word)