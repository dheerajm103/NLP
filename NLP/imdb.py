import requests                                            # Importing libraries
from bs4 import BeautifulSoup as bs  
import re 
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
#FOR UNIGRAM ********************************************************************************************************** 

idmb_reviews=[]

# loading mitv reviews from idmb
for i in range(1,21):
  ip=[]  
  url="https://www.imdb.com/title/tt9544034/reviews/?ref_=tt_ql_urv"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")
  reviews = soup.find_all("div",attrs={"class","text show-more__control"}) 
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  idmb_reviews=idmb_reviews+ip  

# writng reviews in a text file 
with open("idmb.txt","w",encoding='utf8') as output:
    output.write(str(idmb_reviews))
	

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(idmb_reviews)


# Removing unwanted symbols by regular expression
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# tokenising words
ip_reviews_words = ip_rev_string.split(" ")

#TFIDF for weights

vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 1))
X = vectorizer.fit_transform(ip_reviews_words)

# taking stop words
with open("stop.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["family","man","manoj","bajpayee","series","web","bajpai","indian","members","sumeet","kotian","sacred","games","raj","dk"])

# obtaining words witout stop words
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# calculating frequencies for obtained words

fdist = FreqDist(ip_reviews_words )
print(fdist)

# plotting frequencies
fdist.plot(30,cumulative=False)
plt.show()

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud on obtained words

wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# sentiment analysis
# positive words 
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

# word cloud for positive word
wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words 
with open("negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)

# plotting count for positive and negative words
ip_pos_in_pos = ip_pos_in_pos.split(" ")
positive = list(set(ip_pos_in_pos))
ip_neg_in_neg = ip_neg_in_neg.split(" ")
negative = list(set(ip_neg_in_neg))
totaolwords = list(set(ip_reviews_words))
words = ["positive","negative","neutral"]
count = [92,42,526]
ypos = np.arange(len(words))

plt.xticks(ypos,words)
plt.bar(ypos,count)

# BIGRAM*******************************************************************************************************************
WNL = nltk.WordNetLemmatizer()

# Lowercase 
text = ip_rev_string.lower()

# Remove single quote 
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars 
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price','great'] 

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud for bigram
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# for sentiment analysis for bigram
items = list([i[0] for i in words_freq[:951]])
X = pd.DataFrame(columns = ["items"]) 
X["items"] = items
X['polarity'] = X['items'].apply(lambda x : TextBlob(x).polarity)
X['polarity']


plt.hist(X.polarity) ; plt.xlabel("polarity"); plt.ylabel("count")   
    
