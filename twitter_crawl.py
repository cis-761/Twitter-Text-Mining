# Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# locations to be tracked, Capital cities of Midwestern states.
# Topeka, KS
# Indianapolis, IN
# Springfield, IL
# Des Moines, IA
# Lansing, MI
# St. Paul, MN
# Jefferson City, MO
# Lincoln, NE
# Bismarck, ND
# Columbus, OH
# Pierre, SD
# Madison, WI

topeka = [-95.69055556, 38.96333333, -95.56138889, 39.08333333]
indianapolis = [-86.33277778, 39.64638889, -85.99416667, 39.86277778]
springfield = [-89.69833333, 39.67277778, -89.56694444, 39.81416667]
des_moines = [-93.76194444, 41.48583333, -93.545, 41.665]
lansing = [-84.74194444, 42.6275, -84.35027778, 42.89055556]
st_paul = [-93.31, 44.83361111, -92.99833333, 45.01277778]
jefferson_city = [-92.30277778, 38.44555556, -92.00944444, 38.71888889]
lincoln = [-96.845, 40.73944444, -96.52888889, 40.91444444]
bismarck = [-100.9130556, 46.61694444, -100.5311111, 46.82055556]
columbus = [-83.07805556, 39.87805556, -82.86916667, 40.02666667]
pierre = [-100.665, 44.24444444, -99.98833333, 44.51638889]
madison = [-89.53555556, 42.98972222, -89.27611111, 43.13583333]

locations_combined = topeka + indianapolis + springfield + des_moines + lansing + \
    st_paul + jefferson_city + lincoln + bismarck + columbus + pierre + madison

tokens = []
f = open("secret.txt", "r")
lines = list(f)
for x in lines:
    tokens.append(x.strip())
    
f.close()

# Variables that contains the user credentials to access Twitter API
access_token = tokens[0]
access_token_secret = tokens[1]
consumer_key = tokens[2]
consumer_secret = tokens[3]

# This is a basic listener that just prints received tweets to stdout.

class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)
        return False # cuts the connection if an error occurs


if __name__ == '__main__':

    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    # This line filter Twitter Streams to capture data by the keywords: provided below OR form these locations
    stream.filter(track=['flu', 'flushot', 'sick', 'ill', 'avian flu', 'bunged up', 'catarrh', 'chesty', 'chill', 'cold', 'congested', 'cough', 'coughing fit',
                         'frog', 'hacking cough', 'head cold', 'influenza', 'runny', 'sneeze', 'sniff', 'sniffle', 'swine flu', 'common cold'], locations=locations_combined)
