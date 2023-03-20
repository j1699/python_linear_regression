import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df.head()

x = df[['chest_size', 'waist_measurement', 'crotch_height', 'head_circumference']]
y = df[['weight']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("chest_size & waist_measurement")
plt.ylabel("weight")
plt.title("weight_predict")
plt.show()

print(mlr.coef_)
print(mlr.score(x_train, y_train))

while 1:
    chest_size = float(input("가슴 줄레를 입력해 주세요.(0 입력 시 종료)"))
    if chest_size == 0:
        break
    else:
        waist_measurement = float(input("허리 둘레를 입력해 주세요.(0 입력 시 종료)"))
        if waist_measurement == 0:
            break
        else:
            crotch_height = float(input("샅높이를 입력해 주세요.(0 입력 시 종료)"))
            if crotch_height == 0:
                break
            else:
                head_circumference = float(input("머리 둘레를 입력해 주세요.(0 입력 시 종료)"))
                if head_circumference == 0:
                    break
                else:
                    print("예측 몸무게 : " + str(mlr.predict([[chest_size, waist_measurement, crotch_height, head_circumference]])))