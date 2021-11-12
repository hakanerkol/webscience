# HOW TO RUN THE ANALYSIS ON DATA PROVIDED
1. Run the runCluster file to create the graphs, tables and statistics for the tweets.json file. 

# HOW TO RUN THE CRAWLER TO COLLECT NEW DATA
1. Place your Twitter API credentials in the createCreds.py script.
2. Run the createCreds.py script, and it will create the credentials file for Twitter API use.

# The files: 

StreamListener.py and ReadMongo.py are classes used in the other scripts whose
names start with either "run" or "export". 

exportMongoDB.py - Exports MongoDB database to tweets.json.

ReadMongo.py - Contains one function "query" to query the MongoDB, and return the result(s)
in a list. This is used in a few of the scripts.

runListener.py - Gathers the data from the Twitter API and stores it in the MongoDB.

runReader.py - Used for testing to retrieve all unique usernames from MongoDB tweets database.

StreamListener.py - Class used to listen to Twitter Streaming API.

runCluster.py and runCluster.ipynb - Same file. Easier way to run this would be to use a 
Jupyter notebook with the .ipynb file.

