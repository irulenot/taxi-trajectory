import pandas as pd
import ast
import numpy as np
from pandas.api.types import is_numeric_dtype

def main():
    # Shorten data to 10,000 data points
    df = pd.read_csv('data/train.csv')
    df_shortened = df.sample(n=10000, random_state=42)
    df_shortened.to_csv('data/train_shortened.csv', index=False)

    # Format data for future use
    df = pd.read_csv('data/train_shortened.csv')
    df.fillna(0, inplace=True)
    for column in df:
        if column == 'POLYLINE':
            df[column] = df[column].apply(ast.literal_eval)
        elif column == 'MISSING_DATA':
            df[column] = df[column].astype(float).replace({True: 1.0, False: 0.0})
        elif not is_numeric_dtype(df[column]):
            df[column] = pd.factorize(df[column])[0]
    coordinates = pd.DataFrame(df['POLYLINE'].apply(np.ravel))
    coordinates_X = coordinates.copy()
    last_two = coordinates['POLYLINE'].apply(lambda x: list(x)[-2:])
    Y_values = pd.DataFrame(last_two.tolist(), columns=['second_last', 'last'])
    Y_values.to_csv('data/train_shortened_Y.csv', index=False)

    # Add padding to coordinates
    lengths = coordinates['POLYLINE'].apply(lambda x: len(x))
    max_length = max(lengths)
    def pad_array(arr):
        if len(arr) < max_length:
            padding = max_length - len(arr)
            arr = np.pad(arr, (0, padding), mode='constant', constant_values=0)
        return arr
    coordinates['POLYLINE'] = coordinates['POLYLINE'].apply(pad_array)

    coordinates_X['POLYLINE'] = coordinates_X['POLYLINE'].apply(lambda x: x[:-2])  # Removes last two destination coordinates
    lengths_X = coordinates_X['POLYLINE'].apply(lambda x: len(x))
    max_length_X = max(lengths_X)
    def pad_array_X(arr):
        if len(arr) < max_length_X:
            padding = max_length_X - len(arr)
            arr = np.pad(arr, (0, padding), mode='constant', constant_values=0)
        return arr
    coordinates_X['POLYLINE'] = coordinates_X['POLYLINE'].apply(pad_array_X)

    # Map coordinate values to new columns
    df = df.drop('POLYLINE', axis=1)

    coordinates.to_numpy()
    coordinates = coordinates['POLYLINE'].apply(lambda x: pd.Series(x))
    df_full = pd.concat([df, coordinates], axis=1)
    df_full.to_csv('data/train_shortened_full.csv', index=False)

    coordinates_X.to_numpy()
    coordinates_X = coordinates_X['POLYLINE'].apply(lambda x: pd.Series(x))
    df_X = pd.concat([df, coordinates_X], axis=1)
    df_X.to_csv('data/train_shortened_X.csv', index=False)


if __name__ == "__main__":
    main()