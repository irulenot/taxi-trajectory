# Data source: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

import pandas as pd

def main():
    df = pd.read_csv('data/train.csv')
    print('Data shape:', df.shape)
    print('First 5 rows of data:')
    print(df.head())
    print('Summary statistics:')
    print(df.describe())
    print('Data information:')
    print(df.info())

if __name__ == "__main__":
    main()