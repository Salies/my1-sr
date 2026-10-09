from pysr import PySRRegressor
import pandas as pd

df = pd.read_csv("test_data_train.csv")
# tira coluna de dia
df = df.drop(columns=['day'])

df_ozone = df.drop(columns=['pm25'])
df_pm25 = df.drop(columns=['ozone'])

X_ozone = df_ozone.drop(columns=['ozone'])
y_ozone = df_ozone['ozone']
X_pm25 = df_pm25.drop(columns=['pm25'])
y_pm25 = df_pm25['pm25']

# ozônio

model_ozone = PySRRegressor(
    maxsize=30,
    model_selection="best",
    population_size=100,
    parsimony=0.001,
    niterations=1_000,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "exp",
        "log",
        "abs",
        "sin",
        "cos",
        "sqrt"
    ],
    output_directory="out_ozone"
)

print("Treinando ozônio...")
model_ozone.fit(X_ozone, y_ozone)

# pm25

model_pm25 = PySRRegressor(
    maxsize=30,
    model_selection="best",
    population_size=100,
    parsimony=0.001,
    niterations=1_000,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "exp",
        "log",
        "abs",
        "sin",
        "cos",
        "sqrt"
    ],
    output_directory="out_pm25"
)

print("Treinando pm25...")
model_pm25.fit(X_pm25, y_pm25)
