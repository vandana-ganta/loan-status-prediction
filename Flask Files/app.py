from flask import Flask, jsonify,render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    refno = [int(x) for x in request.form.values()] #.to_dict() method is used to convert a dataframe into a dictionary of series or list
    refno = [np.array(refno)]
    result = model.predict(refno)

    if result[0]== 1:
        prediction ='Congratulations!Status for your Loan is Approved,Go visit nearby bank and apply for loan'
    else:
        prediction ='Sorry! Status for your Loan Application is rejected'
    return render_template("index.html", prediction = prediction)
if __name__ == "__main__":

    app.run(debug=True)
