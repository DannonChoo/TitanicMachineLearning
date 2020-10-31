#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  features = [int(x) for x in request.form.getlist('optionId')]

  print('Features: ',features)
  
  if features[2] == 0:
      features.insert(3, 0)
      features.insert(5, 1)
  else:
      features.insert(3, 1)
      features.insert(5, 0)
    
  final_features = [np.array(features)]

  pred = model.predict(final_features)
  
  if pred == 0:
      output = "You would not have survived the Titanic"
  else:
      output = "You are likely to survive the titanic"
  
  print(output)
    
  return render_template('index.html', prediction_text = output)
  return redirect('/')


if __name__ == "__main__":
  app.run(debug=True)


# In[ ]:




