import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train_path = sys.argv[1]
test_path = sys.argv[2]
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

def fill_missing(df):
  lux = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
  categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

  df['Age'] = df['Age'].fillna(df['Age'].mean())
  df['VIP'] = df['VIP'].fillna(df['VIP'].mode())

  for col in categorical:
    most_freq = df[col].value_counts(dropna=True).idxmax()
    df[col] = df[col].fillna(most_freq)

  for col in lux:
    df[col] = df[col].fillna(df.groupby('VIP')[col].transform('median'))

  return df;

def extract_deck(cabin):
  if pd.isna(cabin):
    return cabin
  else:
    return cabin.split('/')[0]

def extract_side(cabin):
  if pd.isna(cabin):
    return cabin
  else:
    return cabin.split('/')[2]

def extract_group(id):
  return id.split('_')[0]

def find_cabin(df):
  df['Group'] = df['PassengerId'].apply(extract_group)
  g_cabin = (df.groupby(['Group']+['Cabin']).size()
  .to_frame('counts').reset_index()
  .sort_values('counts', ascending=False)
  .drop_duplicates(subset='Group')).drop(columns='counts')
  df.loc[df.Cabin.isnull(), 'Cabin'] = df.Group.map(g_cabin.set_index('Group').Cabin)
  return df

def fill_cabin(df):
  cabin = ['Deck', 'Side']
  for col in cabin:
    most_freq = df[col].value_counts(dropna=True).idxmax()
    df[col] = df[col].fillna(most_freq)
  return df

def drop_cols(df):
  cols = ['PassengerId', 'FoodCourt', 'Cabin', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck', 'Name','Group']
  for column in cols:
        df = df.drop(column, axis = 1)
  return df
  
def to_int(df):
  df = df.astype({'Age' : 'int'})
  df = df.astype({'Luxury' : 'int'})
  return df;

def data_preproc(df):
  df = fill_missing(df)
  df = find_cabin(df)
  df['Luxury'] = df['FoodCourt'] + df['RoomService'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
  df['Deck'] = df['Cabin'].apply(extract_deck)
  df['Side'] = df['Cabin'].apply(extract_side)
  df = fill_cabin(df)
  df = to_int(df)
  df = drop_cols(df)
  return df

train_id = train.loc[:, 'PassengerId']
test_id = test.loc[:, 'PassengerId']
train_data = data_preproc(train)
train_data = train_data.drop('Transported', axis=1)
train_target = train.loc[:,'Transported']
test_data = data_preproc(test)

categorical = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
le = LabelEncoder()
for col in categorical:
  le = le.fit(train_data[col])
  train_data[col] = le.transform(train[col])
  test_data[col] = le.transform(test[col])

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_data, train_target)
pred = pd.Series(clf.predict(test_data))

result = pd.DataFrame({
    'PassengerId' : test_id,
    'Transported' : pred
})
result.to_csv('1810052.csv', index=False)
