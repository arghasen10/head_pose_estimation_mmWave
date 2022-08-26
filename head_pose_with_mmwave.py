import pandas as pd
import os
import matplotlib.pyplot as plt
import json

df_lists = []
file_path = '/home/argha/Documents/nexardata/processed/anirban/final_processed/'
csv_files = []
for files in os.listdir(file_path):
    if files.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path + files)
        df_lists.append(df)

anirban_df = pd.concat(df_lists, axis=0, ignore_index=True)
print(anirban_df)

plt.plot(anirban_df['x'], label='x')
plt.plot(anirban_df['y'], label='y')
plt.plot(anirban_df['z'], label='z')
plt.legend()
plt.show()

df_lists = []
file_path = '/home/argha/Documents/nexardata/processed/sugandh/final_processed/'
csv_files = []
for files in os.listdir(file_path):
    if files.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path + files)
        df_lists.append(df)

sugandh_df = pd.concat(df_lists, axis=0, ignore_index=True)
print(sugandh_df)

plt.plot(sugandh_df['x'], label='x')
plt.plot(sugandh_df['y'], label='y')
plt.plot(sugandh_df['z'], label='z')
plt.legend()
plt.show()

mmwave_file_path = '/home/argha/Documents/driver-head-pose/'
files_mmawave = os.listdir(mmwave_file_path)

anirban_files = []
sugandh_files = []

for file in files_mmawave:
    if '.txt' in file:
        if 'anriban' in file:
            anirban_files.append(mmwave_file_path + file)
        elif 'sugandh' in file:
            sugandh_files.append(mmwave_file_path + file)

anirban_mmwave_df = []
sugandh_mmwave_df = []

for file in anirban_files:
    data = [json.loads(val) for val in open(file, "r")]
    df = pd.DataFrame()
    for d in data:
        df = df.append(d['answer'], ignore_index=True)

    anirban_mmwave_df.append(df)

anirban_mmwave_df_final = pd.concat(anirban_mmwave_df, axis=0, ignore_index=True)

for file in sugandh_files:
    data = [json.loads(val) for val in open(file, "r")]
    df = pd.DataFrame()
    for d in data:
        df = df.append(d['answer'], ignore_index=True)

    sugandh_mmwave_df.append(df)

sugandh_mmwave_df_final = pd.concat(sugandh_mmwave_df, axis=0, ignore_index=True)

print(sugandh_mmwave_df_final.head())
