#importing libraries
import numpy as np
import pandas as pd
import pickle
loan=pd.read_csv("Loan.csv",encoding='utf-8')


loan=loan.drop(['Loan_ID'],axis=1)

loan.Gender=loan.Gender.fillna('Male')
loan.Married=loan.Married.fillna('Yes')
loan.Dependents=loan.Dependents.fillna('0')
loan.Self_Employed=loan.Self_Employed.fillna('No')
loan.LoanAmount=loan.LoanAmount.fillna(loan.LoanAmount.mean())
loan.Loan_Amount_Term=loan.Loan_Amount_Term.fillna(loan.Loan_Amount_Term.mean())
loan.Credit_History=loan.Credit_History.fillna('1')
print(loan.head())
loan.Education=loan.Education.replace({'Graduate':1,'Not Graduate':0})
loan.Self_Employed=loan.Self_Employed.replace({'Yes':0,'No':1})
loan.Dependents=loan.Dependents.replace({'3+':4})
loan.Property_Area=loan.Property_Area.replace({'Urban':2,'Rural':0,'Semiurban':1})
loan.Loan_status=loan.Loan_status.replace({'Y':1,'N':0})
loan.Gender=loan.Gender.replace({'Male':1,'Female':0})
loan.Married=loan.Married.replace({'Yes':1,'No':0})


x=loan.iloc[:,0:11].values
y=loan.iloc[:,-1].values


from sklearn.ensemble import RandomForestClassifier
rmf=RandomForestClassifier()
rfoc=rmf.fit(x,y)

# Saving model to disk
pickle.dump(rfoc, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
new_loan_application = [0, 0, 0, 0, 1, 1, 0.0, 405.0, 360.0, 0, 0]
p = model.predict([new_loan_application])
print(p)
