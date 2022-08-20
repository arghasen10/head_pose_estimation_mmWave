import pandas as pd
import os
import time
import numpy as np

global_fps = 3
y_min = -4
y_max = 9
x_min = -6
x_max = 6


def reformat_milli(ser):
    i = 0
    t_list = []
    past_time = ser[0]
    for e in ser:
        t_list.append(e + f"_{i}")
        i += 1
        if e != past_time:
            i = 0
            past_time = e
    return t_list


def process_imu(filename):
    imu_data = pd.read_csv(filename, header=None)
    timestamps = imu_data.iloc[:, 0].values
    time_ARR = []
    for val in timestamps:
        time_val = int(val.split('|')[0]) / 1000000
        time_ARR.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_val - 3)))
    imu_data['datetime'] = time_ARR

    imu_data = imu_data[['datetime', 1]]
    imu_data.columns = ['datetime', 'acc']

    g = imu_data.groupby('datetime')

    ds_list = []
    for name, gr in g:
        stride = int(gr.shape[0] / global_fps)
        ds_gr = pd.DataFrame({'datetime': [name] * global_fps,
                              'acc': [
                                  gr.iloc[i * stride:i * stride + stride].drop(columns=['datetime']).mean().values[0] \
                                  for i in range(global_fps)]
                              })
        ds_list.append(ds_gr)
    imu_data = pd.concat(ds_list, ignore_index=True)
    return imu_data


def find_level(x, y, z):
    global text
    if (y_min < y < y_max) and (x_min < x < x_max):
        text = "looking_forward"
    elif (y > y_max) and (x < x_min):
        text = "looking_down_and_right"
    elif (y > y_max) and (x > x_max):
        text = "looking_up_and_right"
    elif y > y_max:
        text = "looking_right"
    elif (y < y_min) and (x < x_min):
        text = "looking_down_and_left"
    elif (y < y_min) and (x > x_max):
        text = "looking_up_and_left"
    elif y < y_min:
        text = "looking_left"
    elif x < x_min:
        text = "looking_down"
    elif x > x_max:
        text = "looking_up"
    return text


def process_image(filename):
    image_df = pd.read_csv(filename)
    image_df['datetime'] = image_df[['date', 'time']].apply(lambda e: e[0] + ' ' + e[1], axis=1)
    image_df = image_df[['datetime', 'x', 'y', 'z']]
    g = image_df.groupby('datetime')
    ds_list = []
    for name, gr in g:
        stride = int(gr.shape[0] / global_fps)
        vals = np.array([gr.iloc[i * stride:i * stride + stride].drop(columns=['datetime']).mean().values \
                         for i in range(global_fps)])
        ds_gr = pd.DataFrame({'datetime': [name] * global_fps,
                              'x': vals[:, 0], 'y': vals[:, 1], 'z': vals[:, 2]
                              })
        ds_list.append(ds_gr)
    image_df = pd.concat(ds_list, ignore_index=True)
    image_df['level'] = image_df[['x', 'y', 'z']].apply(lambda e: find_level(e[0], e[1], e[2]), axis=1)
    return image_df


anirban_nexar_path = '/home/argha/Documents/nexardata/processed/anirban/final_processed/'
anirban_nexar_files = os.listdir(anirban_nexar_path)
anirban_imu_df_list = []
anirban_image_df_list = []
for file in anirban_nexar_files:
    if 'A.csv' in file:
        df = process_imu(anirban_nexar_path + file)
        df['datetime'] = reformat_milli(df['datetime'])
        anirban_imu_df_list.append(df)
    elif '.csv' in file:
        df = process_image(anirban_nexar_path + file)
        df['datetime'] = reformat_milli(df['datetime'])
        anirban_image_df_list.append(df)

anirban_imu_df = pd.concat(anirban_imu_df_list, axis=0, ignore_index=True)
anirban_image_df = pd.concat(anirban_image_df_list, axis=0, ignore_index=True)

print(anirban_imu_df.shape)
print(anirban_image_df.shape)
print(anirban_imu_df.head())
print(anirban_image_df.head())
