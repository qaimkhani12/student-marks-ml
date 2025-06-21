import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv("students_data.csv")
x=df[["hours"]]
y=df['marks']
model=LinearRegression()
model.fit(x,y)
import joblib
joblib.dump(model,"student_model.pkl")
print("model saved successfully!")
loaded_model=joblib.load("student_model.pkl")
result=loaded_model.predict([[7]])
print("7 ghante ke liye prediction:",result[0])