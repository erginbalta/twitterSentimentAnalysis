from sklearn.model_selection import train_test_split
import dbOperations as db
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

dbTweets = db.getTweets()

train, test = train_test_split(dbTweets, test_size = 0.3)

def tweetCleaner(tweets):
    cleanedTweets = []
    for tweet in tweets:
        tweet1 = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        tweet2 = re.sub('@[^\s]+', 'AT_USER', tweet1)
        tweet3 = re.sub(r'#([^\s]+)', r'\1', tweet2)
        cleanedTweets.append(tweet3)
    return cleanedTweets

def tokanizationTweet(cleanTweets):

    tokenizedTweets = []
    #tokenizedTweets = word_tokenize(cleanTweets, language="english", preserve_line=False)
    for tweet in cleanTweets:
        tokenizedTweet = word_tokenize(str(tweet))
        tokenizedTweets.append(tokenizedTweet)
    return tokenizedTweets

def removeStopword(tokenTweets):
    filteredTweets = []
    for tweet in tokenTweets:
        filterTweet = []
        for word in tweet:
            if word not in stopwords.words("english"):
                filterTweet.append(word)
                
        filteredTweets.append(filterTweet)
    
    return filteredTweets

def posTagTweets(tokenTweets):
    posTagTweet = []
    for tweet in tokenTweets:
        posTagTweet.append(nltk.pos_tag(tweet))
    
    return posTagTweet

def buildVocabulary(preprocessedTrainingData):
    all_words = []
    
    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features

def extract_features(tweet,word_features):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

def sentimentAnalyzer(trainData,testData):
    cleanTweetsTrain = tweetCleaner(trainData)
    tokenizatedTweetTrain = tokanizationTweet(cleanTweetsTrain)
    noStopwordsTweetsTrain = removeStopword(tokenizatedTweetTrain)
    #posTagTweetsTrain = posTagTweets(tokenizatedTweetTrain)
    
    cleanTweetsTest = tweetCleaner(testData)
    tokenizatedTweetTest = tokanizationTweet(cleanTweetsTest)
    noStopwordsTweetsTest = removeStopword(tokenizatedTweetTest)
    #posTagTweetsTest = posTagTweets(tokenizatedTweetTest)
    
    word_features = buildVocabulary(tokenizatedTweetTrain)
    trainingFeatures=nltk.classify.apply_features(extract_features,noStopwordsTweetsTrain)
    
    NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
    
    NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in noStopwordsTweetsTest]
    
    #result = db.prepareMongoDocument(testData,NBResultLabels)
    #db.writeSentimentResult(result)
    
    
    if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
        print("Overall Positive Sentiment")
        print("Positive Sentiment Percentage = " + str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
    else: 
        print("Overall Negative Sentiment")
        print("Negative Sentiment Percentage = " + str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")


sentimentAnalyzer(train,test)