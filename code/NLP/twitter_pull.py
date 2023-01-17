import tweepy
import pandas as pd
ACCESS_TOKEN = ''
ACCESS_SECRET = ''
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
# authenticate
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth)

search = "@glossier"


data = {
	"id" : [],
#	"text" : [],
	"full_text" : [],
	"user_location" : [],
	"geo" : [],
	"coordinates" : [],
	"place" : []
}

for tweet in tweepy.Cursor(api.search_tweets,q=search, lang="en", tweet_mode='extended').items(10000):
	data["id"].append(tweet.id)
#	data["text"].append(tweet.text)
	data["full_text"].append(tweet.full_text)
	data["user_location"].append(tweet.user.location)
	data["geo"].append(tweet.geo)
	data["coordinates"].append(tweet.coordinates)
	data["place"].append(tweet.place)


#tweets = tweepy.Cursor(api.search_tweets,
                   #    q=search,
                    #   lang="en", tweet_mode='extended')
#print(type(tweets))
df = pd.DataFrame(data)
df.to_csv("twitter_data.csv")
