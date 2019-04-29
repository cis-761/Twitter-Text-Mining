import json
import pandas as pd

tweets_data_path = 'data.txt'

tweets_data = []    
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

tweets = pd.DataFrame()    

tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['location'] = list(map(lambda tweet: tweet['coordinates'], tweets_data))
tweets['favorite'] = list(map(lambda tweet: tweet['favorited'], tweets_data))
tweets['date'] = list(map(lambda tweet: tweet['created_at'], tweets_data))

user = pd.DataFrame()




tweets.to_csv('test.csv', index = None, header = True)