import pandas as pd #type: ignore
import os #type: ignore
import numpy as np #type: ignore
import random
import math

fp = os.path.expanduser('~/sampled_10k_df3.csv')
df = pd.read_csv(fp)

total_len = len(df)
threshold = 4096

df['prompt_len'] = df['prompt_len'].astype(int)

filtered_rows = df[df['prompt_len'] > threshold]
filtered_len = len(filtered_rows)

perc = (filtered_len/total_len)*100

print(f"total filtered_rows over 2048: {filtered_len}, percentage: {perc}")
# Step 2: Count the number of rows that satisfy the condition
#count = len(filtered_rows)