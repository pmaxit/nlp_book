import pandas as pd
from sklearn.model_selection import train_test_split
import csv

train = pd.read_csv('./data/toxic/train.csv.zip',engine='python',quoting=csv.QUOTE_ALL)
test_data = pd.read_csv('./data/toxic/test.csv.zip',engine='python',quoting=csv.QUOTE_ALL)
test_labels = pd.read_csv('./data/toxic/test_labels.csv.zip',engine='python')


train, valid = train_test_split(train , test_size=0.2)
train.to_csv("./data/toxic/train_new.csv",index=False)
valid.to_csv("./data/toxic/valid.csv", index=False)


