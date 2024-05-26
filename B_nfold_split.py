import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import os, sys

data = sys.argv[1]
outdir = sys.argv[2]
N = 5

df = pd.read_csv(data)
skf = StratifiedKFold(n_splits=N, random_state=0, shuffle=True)

for i, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
    xTrain = df.loc[train_index, :]
    xTest = df.loc[test_index, :]
    new_xTrain, xDev, _, _ = train_test_split(xTrain, xTrain['label'], test_size=0.2, stratify=xTrain['label'])

    print('train:{} dev:{} test:{}'.format(len(new_xTrain), len(xDev), len(xTest)))

    split_path = '{}/data_splits_{}'.format(outdir, i+1)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    new_xTrain.to_csv(f'{split_path}/train.csv', index=False)
    xDev.to_csv(f'{split_path}/dev.csv', index=False)
    xTest.to_csv(f'{split_path}/test.csv', index=False)

