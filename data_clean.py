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
users['geo_enabled'] = list(map(lambda user: user['geo_enabled'], user_data))
users['verified'] = list(map(lambda user: user['verified'], user_data))

user_frame = users.drop('user', axis=1)

print(users_frame.columns)
print(users_frame.shape)


# This is where we write out the DataFrames to CSV. 
users_frame.to_csv('user.csv', index=None, header=True)
tweets.to_csv('tweets.csv', index = None, header = True)