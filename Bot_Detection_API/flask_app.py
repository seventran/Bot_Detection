from flask import Flask
from flask_celery import make_celery
from config import *
import pandas as pd
import numpy as np
import tweepy
import re
import json
import redis 
import vincent
import dill as pickle
import time

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND
celery = make_celery(app)

@app.route('/process/<name>')
def process(name):
    res = crawl_preprocess.delay(name)
    while (res.status == 'PENDING'):
        continue
    if (res.status == 'SUCCESS'):
        predict.delay(res.id)
        return 'Predicting'
    else:
        return 'Can not crawl data to access'

@celery.task(name='flask_app.crawl_preprocess', bind = True)
def crawl_preprocess(self, user):
    #Get twitter API
    CONSUMER_KEY = 'Tnv8Z7EKKLzPUE8Hhux2Djb6S'
    CONSUMER_SECRET = 'fJJ2EqawJaUvC8UGcghinexcAHdML9wQKMwbQlSmm03a7C0q0m'
    ACCESS_TOKEN = '995091920803151874-cQoPco9WqeE6u2u7rbEeoF5LprUEUte'
    ACCESS_SECRET = 'y9nAyWXaFMe1MBYU8x3ac35mBAHl6NaQqNyxrjLX14sh9'
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    extractor = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        acc = extractor.get_user(user)
        tweets = extractor.user_timeline(screen_name=user, count=3000)
    except tweepy.TweepError:
        self.update_state(state='FAILURE')
        return 'Can not crawl data to access'
    if (len(tweets) >= 100):
        n = 100
    else:
        n = len(tweets)
    #Create tweets'feature
    ment_user_mean = 0
    hashtag_mean = 0
    retweet_mean = 0
    like_mean = 0

    for tweet in tweets[:n]:
        ment_user_mean += len(re.findall(r"@(\w+)", tweet.text))
        hashtag_mean += len(re.findall(r"#(\w+)", tweet.text))
        retweet_mean += tweet.retweet_count
        like_mean += tweet.favorite_count

    ment_user_mean = ment_user_mean/n
    hashtag_mean = hashtag_mean/n
    retweet_mean = retweet_mean/n
    like_mean = like_mean/n
    
    #Create user'features
    listed = acc.listed_count
    favorites = acc.favourites_count
    friends = acc.friends_count
    verified = int(acc.verified)
    default_profile =  int(acc.default_profile)
        

    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                    r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                    r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                    r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

    screen_name = int(pd.Series(acc.screen_name).str.contains(bag_of_words_bot, case=False, na=False))
    name = int(pd.Series(acc.name).str.contains(bag_of_words_bot, case=False, na=False))
    description = int(pd.Series(acc.description).str.contains(bag_of_words_bot, case=False, na=False))

    #features = pd.DataFrame(data = [screen_name, name, description, verified, default_profile, ment_user_mean, hashtag_mean, retweet_mean, like_mean, listed, favorites, friends], dtype = float)
    #return features.to_json(orient="index")
    features = [screen_name, name, description, verified, default_profile, ment_user_mean, hashtag_mean, retweet_mean, like_mean, listed, favorites, friends]
    return json.dumps(features)

@celery.task(name='flask_app.predict')
def predict(id):
    clf = 'model_v1.pk'
    print("Loading the model...")
    loaded_model = None
    with open(clf,'rb') as f:
        loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...")
    vincent.core.initialize_notebook()
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    rd = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    key = 'celery-task-meta-' + id
    data = rd.get(key)
    if data != None:
        features = json.loads(data)
    else:
        return 'Can not access data to evaluate'
    feat_arr = np.array([float(s) for s in features['result'][1:-1].split(', ')])
    print (type(feat_arr))
    print (feat_arr)
    prediction = loaded_model.predict(feat_arr.reshape(1, -1))
    if prediction:
        return "It is a bot"
    else:
        return "It is a user"

if __name__ == '__main__':
    app.run(debug=True)
