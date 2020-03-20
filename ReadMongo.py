import pymongo

def query(find, show={}):
    # Connect to MongoDB and insert the tweet to the database
    client = pymongo.MongoClient('127.0.0.1', 27017)
    db = client.tweets # Database name = 'tweets'
    col = db.streamed_tweets # Collection name = 'streamed_tweets'
    results = []
    for x in col.find(find, show):
        results.append(x)
    return results
