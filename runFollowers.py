from twython import Twython
import ReadMongo
import pymongo
import json
import time
credentials = {}
try:
    # Load credentials
    with open("twitter_credentials.json", "r") as file:
        credentials = json.load(file)
        if len(credentials.keys()) == 0:
            # Empty credentials file
            exit("Please add credentials to credentials file")
except:
    # Failed to load credentials, exit.
    exit("Failed to load credentials")

twitter = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'],
                    credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])

def getUsernames():
  results = ReadMongo.query({}, {'user':1, '_id': 0})
  return list(set([x['user'].encode('ascii') for x in results]))

def save_to_mongo(data):
  print(data)
  return
  # Connect to MongoDB and insert the tweet to the database
  client = pymongo.MongoClient('127.0.0.1', 27017)
  db = client.users # Database name = 'users'
  col = db.user_following # Collection name = 'streamed_tweets'
  col.upsert(data)

users = getUsernames()
for user in users:
  followers = twitter.get_followers_ids(screen_name=user)
  print(followers)
  time.sleep(1)
