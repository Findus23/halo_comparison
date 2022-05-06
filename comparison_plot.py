import numpy as np
import pandas as pd

df = pd.read_csv("shannon_128_100_shannon_512_100.csv")

bins = np.geomspace(450, 61000, 10)
print(bins)

digits = np.digitize(df.ref_masses, bins)
print(digits)

for i, bin in enumerate(bins):
    values = np.where(digits == i)
    in_bin = df.iloc[values]
    matches = np.array(in_bin.matches)
    print(matches.mean())
