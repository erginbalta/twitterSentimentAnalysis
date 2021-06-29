# -*- coding: utf-8 -*-
import pymongo

mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
tweetDb = mongoClient["tweets"]
tweetColl = tweetDb["trumpTweets"]

def getTweets():
    tweets=[]
    dbResult = tweetColl.find({},{"_id":0,"id":0,"link":0,"date":0,"retweets":0,"favorites":0,"mentions":0,"hashtags":0})
    for tweet in dbResult:
        tweets.append(tweet["content"])


    return tweets

