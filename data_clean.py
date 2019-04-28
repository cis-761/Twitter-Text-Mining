# JSON -> CSV WITH DATA THAT CORRESPONDS WITH OUR DATABASE. EASIEST IS TO HAVE ONE CSV PER TABLE.



import json
import pandas as pd
import csv


tweets_data_path = 'data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue


# print ( len(tweets_data) )


tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['location'] = map(lambda tweet: tweet['location'], tweets_data)
tweets['date'] = map(lambda tweet: tweet['created_at'], tweets_data)



#for key, val in tweets.items():
    #print (list(map(key, val)))




tweets.to_csv(r'/home/riddy/Downloads/Twitter-Text-Mining-master/tweets.csv', index = None, header = True)
