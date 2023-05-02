# taxi-trajectory

SETUP
-
1. Download data from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
2. Put all CSV files in data directory
3. Run data_explore.py to confirm data is in right location
4. Run data_prep.py to create data for modeling

DATA 
- 
- train_shortened.csv: 10,000 raw data points from train.csv
- train_shortened_full.csv: Processed data from train_shortened.csv
- train_shortened_X.csv: Processed data without destination coordinates
- train_shortened_Y.csv: Processed data destination coordinates