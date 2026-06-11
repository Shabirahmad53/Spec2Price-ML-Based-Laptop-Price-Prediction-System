import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load the dataset
df = pd.read_csv('laptop_price.csv')

# Data Cleaning
# Remove the first column (index)
df.drop(columns=['Unnamed: 0'], inplace=True)

# Clean RAM and Weight
df['Ram'] = df['Ram'].str.replace('GB', '').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

# Feature Engineering
# Extract Resolution and PPI
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Extract X and Y resolution
new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0].str.extract(r'(\d+)').astype('int')
df['Y_res'] = new[1].astype('int')

# Calculate PPI
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float')
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Extract CPU Name
def fetch_processor(text):
    words = text.split()
    if words[0:3] == ['Intel', 'Core', 'i7'] or words[0:3] == ['Intel', 'Core', 'i5'] or words[0:3] == ['Intel', 'Core', 'i3']:
        return ' '.join(words[0:3])
    else:
        if words[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['Cpu brand'] = df['Cpu'].apply(fetch_processor)
df.drop(columns=['Cpu'], inplace=True)

# Extract Memory info
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = new[0].str.strip()
df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.extract(r'(\d+)').astype(int)

df["second"] = new[1].fillna("0")
df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.extract(r'(\d+)').astype(int)

df["HDD"] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df["SSD"] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage', 'Memory'], inplace=True)

# Extract GPU Brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)

# Simplify OS
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)

# Final Train-Test Split
X = df.drop(columns=['Price'])
y = np.log(df['Price']) # Using log of price for better regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Column Transformer for OneHotEncoding
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('R2 score', r2_score(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))

# Save the model and the dataframe (for dropdowns in UI)
pickle.dump(pipe, open('model.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))

print("Model and dataframe pickle files created successfully.")
