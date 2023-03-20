from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("h-w.csv")
x = df["height"]
y = df["weight"]

line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1, 1), y)

plt.plot(x, y, 'o')
plt.plot(x,line_fitter.predict(x.values.reshape(-1,1)))
plt.show()
"""
while 1:
    height = int(input("키를 입력해 주세요.(e로 종료)"))
    if height == "e":
        break
    else:
        print(line_fitter.predict([[height]]))
"""