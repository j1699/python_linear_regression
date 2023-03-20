from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("가상 데이터_업무시설_3.7_제출.csv")
list = df.columns.tolist()
list.remove("Date")

for i in list:
    for j in list:
        x = df[[i]] # i 값을 받는 부분에서 숫자형으로 타입이 설정되어 있음
        y = df[j]
        
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y)
        #R = model.score(x, y)
        
        plt.plot(x, y, 'o')
        plt.plot(x,model.predict(x.values.reshape(-1,1)))
        plt.xlabel(i)
        plt.ylabel(j)
        #plt.title("relation_square: ", R)
        plt.show()