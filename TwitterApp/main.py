import tweepy
from textblob import TextBlob
import csv

#Twitter LogIn
consumer_key = "..."
consumer_secret = "..."
access_token = "...-..."
access_token_secret = "..."

#Settup the API
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

#Input for Search
searchWord = input("Enter a Word to analyze: ")
puplic_tweets = api.search(searchWord,count=200)

#index, tweet, polarity ,  subjectivity
countList = 1
tweetsConverted = []
avgPolarity = float(0)
for tweet in puplic_tweets:
    tmp = TextBlob(tweet.text)
    tweetsConverted.append([countList,tweet.text.encode("utf-8"),tmp.sentiment.polarity,tmp.sentiment.subjectivity])
    avgPolarity = avgPolarity + tmp.sentiment.polarity
    countList+=1
#overall polarity
avgPolarity = float(avgPolarity/countList)

#Create and write the File for CSV
with open('TweetList.csv','w') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'Tweet', 'polarity', 'subjectivity'])
    writer.writerows(tweetsConverted)
    writer.writerow(["Overall Polarity for the Word %s is: %f" % (searchWord,avgPolarity)])


#Itterating threw the tweet-list
for tweet in tweetsConverted:
    print(tweet)
print("Overall Polarity for the Word %s is: %f" % (searchWord,avgPolarity))