
# coding: utf-8

# In[13]:


import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import dill as pickle


# In[16]:


app = Flask(__name__)

@app.route('/predict')
def apicall():
    try:
        main_file_path = 'test_data.csv'
        df = pd.read_csv(main_file_path)
        clf = 'model_v1.pk'
    except Exception as e:
        raise e
        
    name = str(input("Input screen name: "))
    account_feature = df[df['Screen name'] == name]
    account_feature = account_feature.iloc[:,1:-1]
    #Load the saved model
    print("Loading the model...")
    loaded_model = None
    with open(clf,'rb') as f:
        loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...")
    prediction = loaded_model.predict(account_feature)
    if prediction:
        print ("It's a bot")
    else:
        print ("It's a user")
    return (responses)
if __name__ == '__main__':
    app.run(debug = True)

