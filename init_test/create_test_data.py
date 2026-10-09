import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("../data/silver/my1_cleaned.csv")

# convertendo a vel. do vento de knots para m/s
df['wind_speed'] = df['wind_speed'] * 0.514444

# decompondo o vento em componentes u (leste) e v (norte)
theta = np.radians(df['wind_direction'])
df['wind_u'] = -df['wind_speed'] * np.sin(theta)
df['wind_v'] = -df['wind_speed'] * np.cos(theta)

df = df.drop(columns=['wind_speed', 'wind_direction'])

# criando coluna de dia
df['day'] = pd.to_datetime(df['datetime']).dt.date
df = df.drop(columns=['datetime'])

# média diária de cada variável
df = df.groupby('day').mean().reset_index()

# dia julaino
df['julian_day'] = pd.to_datetime(df['day']).dt.dayofyear
# dia da semana
df['weekday'] = pd.to_datetime(df['day']).dt.weekday

# julian_continuous = sin(2pij / 365) + cos(2pij / 365)
df['julian_continuous'] = np.sin(2 * np.pi * df['julian_day'] / 365) + np.cos(2 * np.pi * df['julian_day'] / 365)

# dow_continuous = cos(1.4pid/14) + sin(1.2pid/7)
df['dow_continuous'] = np.cos(1.4 * np.pi * df['weekday'] / 14) + np.sin(1.2 * np.pi * df['weekday'] / 7)

df = df.drop(columns=['julian_day', 'weekday'])

# variávei continau de tempo
df['t'] = (df['day'] - df['day'].min()) / (df['day'].max() - df['day'].min())

print(df.head())

train_df, test_df = train_test_split(df, test_size=0.25, random_state=6661)

print(train_df.shape)
print(test_df.shape)

train_df.to_csv("test_data_train.csv", index=False)
test_df.to_csv("test_data_test.csv", index=False)

