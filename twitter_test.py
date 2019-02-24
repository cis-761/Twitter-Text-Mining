#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

tokens = []
f = open("secret.txt", "r")

for x in f:
    tokens.append(f.readline())
    
f.close()

#Variables that contains the user credentials to access Twitter API 
access_token = tokens[0]
access_token_secret = tokens[1]
consumer_key = tokens[2]
consumer_secret = tokens[3]


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: provided below
    stream.filter(track=['flu', 'flushot'])
    