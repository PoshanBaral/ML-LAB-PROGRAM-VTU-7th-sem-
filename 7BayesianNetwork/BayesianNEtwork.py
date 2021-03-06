import numpy as np
import pandas as pd
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

#Read the attributes
lines = list(csv.reader(open('Heart.csv', 'r')));
attributes = lines[0]
#attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', #	'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']

heartDisease = pd.read_csv('C:\Users\Poshan Baral\Desktop\ML-LAB-PROGRAM-vtu\7BayesianNetwork\Heart.csv', names = attributes)
heartDisease = heartDisease.replace('?', np.nan)

print('Few examples from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes') 
print(heartDisease.dtypes)

model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'),
('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])

print('\nLearning CPDs using Maximum Likelihood Estimators...');
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:') 
HeartDisease_infer = VariableElimination(model)

print('\n1.Probability of HeartDisease given Age=20')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q['heartdisease'])

print('\n2. Probability of HeartDisease given chol (Cholestoral) =100')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100}) 
print(q['heartdisease'])
