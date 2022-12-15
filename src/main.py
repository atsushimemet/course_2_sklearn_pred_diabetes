import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("./data/input/diabetes.csv")

def main():
    df.head()
    df.shape
    df.info()
    df.describe()

# preprocessing
    y = df["Y"].values.reshape(-1, 1)
    x = df["BMI"].values.reshape(-1, 1)
    x.shape
    train_x, train_y, test_x, test_y = (
        x[:-20],
        y[:-20],
        x[-20:],
        y[-20:],
)
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    mse = mean_squared_error(test_y, y_pred)
    mse
    r2_score(test_y, y_pred)
    
    plt.scatter(test_x, test_y, marker="o")
    plt.plot(test_x, y_pred, linestyle="dashdot", marker="o")
    plt.show()
    dirname = './data/output/'
    filename = dirname + 'img.png'
    plt.savefig(filename)
    print('end')


if __name__ == "__main__":
    main()