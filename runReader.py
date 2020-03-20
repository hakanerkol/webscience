import ReadMongo


def getUsernames():
    results = ReadMongo.query({}, {'user':1, '_id': 0})
    return list(set([x['user'].encode('ascii') for x in results]))
print(getUsernames())
print(len(getUsernames()))
