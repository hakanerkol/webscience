import json
from StreamListener import StreamListener

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


# Instantiate StreamListener class with API credentials
stream = StreamListener(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'],
                    credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
# Start the stream
# Track these keywords, since the US Presidential nominees are popular right now post-Super Tuesday
stream.statuses.filter(track='vote, super tuesday, president')
