import json
import pandas as pd
import ast

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

# getting RT Column
retweet_list = []
for k, v in tweets.text.iteritems():
    if v[:2] == "RT":
        retweet_list.append(True)
    else:
        retweet_list.append(False)

tweets['rt'] = retweet_list # this should make this extra column ? 

# Cleaning location 
location_list = []
for k, v in tweets.location.iteritems():
    if v is None:
        location_list.append("")
    else:
        location_list.append(v['coordinates'])

tweets.drop(labels='location', axis='columns', inplace=True)
tweets['location'] = location_list


# start processing users

users = pd.DataFrame()

users['user'] = list(map(lambda tweet: tweet['user'], tweets_data))

users['user'] = users['user'].astype(str)

user_data = [] 

for i in users.user:
    data = ast.literal_eval(i)
    s1 = json.dumps(data)
    user = json.loads(s1)
    user_data.append(user)

# user information we need to collect 
# name, screen_name, geo_enabled, verified

users['name'] = list(map(lambda user: user['name'], user_data))
users['screen_name'] = list(map(lambda user: user['screen_name'], user_data))
users['geo_enabled'] = list(map(lambda user: user['geo_enabled'], user_data)) #clean out as string-coordinates not null, those null stay null
users['verified'] = list(map(lambda user: user['verified'], user_data))

users_frame = users.drop('user', axis=1)

# This is where we write out the DataFrames to CSV. 
users_frame.to_csv('user.csv', index=None, header=True)
tweets.to_csv('tweets.csv', index = None, header = True)