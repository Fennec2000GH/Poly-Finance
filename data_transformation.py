from os import sep
import pandas as pd
from pprint import pprint

df = pd.read_csv('sample data/2011_Retail_Sales_By_Town_All_NAICS_-_ARCHIVE.csv', sep=',')
pprint(df.head())

df['Town'] = df['Municipality'].transform(func=lambda x: x.split()[0].capitalize())
col = column=df.pop(item='Town')
df.insert(loc=0, column='Town', value=col)
pprint(df.head())

df.to_csv(path_or_buf='sample data/2011_Retail_Sales_By_Town_All_NAICS_-_ARCHIVE.csv')