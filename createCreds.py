import json

# Enter keys/secrets as strings in the following fields
credentials = {}
credentials['CONSUMER_KEY'] = ''
credentials['CONSUMER_SECRET'] = ''
credentials['ACCESS_TOKEN'] = ''
credentials['ACCESS_SECRET'] = ''

# Save the credentials object to file
with open("twitter_credentials.json", "w") as file:
    json.dump(credentials, file)
