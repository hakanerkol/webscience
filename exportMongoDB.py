import ReadMongo
import json

def get_tweets_from_mongo():
	return ReadMongo.query({}, {'_id': 0})
	

tweets = {'tweets' : get_tweets_from_mongo() }

# Save the JSON object to file
with open("tweets.json", "w") as file:
    json.dump(tweets, file)
