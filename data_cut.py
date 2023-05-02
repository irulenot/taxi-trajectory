# Data source: https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data

import pandas as pd

def main():
    df = pd.read_csv('data/train.csv')
    df_shortened = df.sample(n=10000, random_state=42)
    print('Shortened data shape:', df_shortened.shape)
    df_shortened.to_csv('data/train_shortened.csv', index=False)

if __name__ == "__main__":
    main()