from twython import TwythonStreamer
import pymongo

class StreamListener(TwythonStreamer):

    def process_tweet(self, tweet):
        # Filter out unwanted data, keeping the important stuff
        d = {}
        d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
        d['text'] = tweet['text']
        d['user'] = tweet['user']['screen_name']
        d['user_loc'] = tweet['user']['location']
        return d

    def on_success(self, data):
        # API call successful
        # Only collect tweets in English
        if 'lang' in data.keys():
            # Avoid key errors if language isn't set
            if data['lang'] == 'en':
                # Process tweet then save to MongoDB
                tweet_data = self.process_tweet(data)
                self.save_to_mongo(tweet_data)

    def on_error(self, status_code, data):
        # If an error occurs, print the issue, and exit
        print(status_code, data)
        self.disconnect()

    def save_to_mongo(self, tweet):
        # Connect to MongoDB and insert the tweet to the database
        client = pymongo.MongoClient('127.0.0.1', 27017)
        db = client.tweets # Database name = 'tweets'
        col = db.streamed_tweets # Collection name = 'streamed_tweets'
        col.insert_one(tweet)
        print(tweet) # Display tweet after saving to database
