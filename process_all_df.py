import argparse

import pandas as pd
import glob
import time
import numpy as np
import json

global_fps = 3
name = 'sugandh'
y_min = -4
y_max = 9
x_min = -6
x_max = 6


def reformat_milli(df):
    df['datetime'] = [ts + f'_{i}' for ts, e in df.groupby('datetime').count().iloc[:, 0].to_dict().items() for i in
                      range(e)]
    return df


def read_imu(filename):
    imu_data = pd.read_csv(filename, header=None)
    timestamps = imu_data.iloc[:, 0].values
    time_ARR = []
    for val in timestamps:
        time_val = int(val.split('|')[0]) / 1000000
        time_ARR.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_val - 3)))
    imu_data['datetime'] = time_ARR

    imu_data = imu_data[['datetime', 1]]
    imu_data.columns = ['datetime', 'acc']
    return imu_data


def process_imu(imu_data):
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
    imu_data = pd.concat(ds_list, ignore_index=True).dropna()
    return (imu_data)


def find_level(x, y, z, x_min, x_max, y_min, y_max=9):
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


def read_image(filename):
    image_df = pd.read_csv(filename)
    image_df['datetime'] = image_df[['date', 'time']].apply(lambda e: e[0] + ' ' + e[1], axis=1)
    image_df = image_df[['datetime', 'x', 'y', 'z']]
    return image_df


def process_image(image_df, x_min, x_max, y_min):
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
    image_df['level'] = image_df[['x', 'y', 'z']].apply(lambda e: find_level(e[0], e[1], e[2], x_min, x_max, y_min), axis=1)
    return image_df.dropna()


def process_mmWave(filename):
    data = [json.loads(val) for val in open(filename, "r")]
    mmwave_df = pd.DataFrame()
    for d in data:
        mmwave_df = mmwave_df.append(d['answer'], ignore_index=True)

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: '2022-08-08 ' + ':'.join(e.split('_')))
    mmwave_df = mmwave_df[['datetime', 'x_coord', 'y_coord', 'z_coord', 'rp_y', 'doppz']]
    return mmwave_df.dropna()


def get_final_df(name, x_min, x_max, y_min):
    nexar_path = f'/home/argha/Documents/nexardata/processed/{name}/'
    mmwave_path = '/home/argha/Documents/driver-head-pose/'
    image_path = f'/home/argha/Documents/nexardata/processed/{name}/final_processed/'

    imu_data = reformat_milli(process_imu(pd.concat([read_imu(f) for f in glob.glob(nexar_path + '**A.dat')])))
    mmwave_data = pd.concat([reformat_milli(process_mmWave(f)) for f in glob.glob(mmwave_path + f'**{name}_drive.txt')])
    image_df = reformat_milli(process_image(pd.concat([read_image(f) for f in glob.glob(image_path + '**.csv')]), x_min, x_max, y_min))

    image_df.set_index('datetime', inplace=True)
    imu_data.set_index('datetime', inplace=True)
    mmwave_data.set_index('datetime', inplace=True)

    processed = pd.concat([image_df, imu_data, mmwave_data], join='inner', axis=1).reset_index()
    return processed


def parseArg():
    parser = argparse.ArgumentParser(description='Collect dataframe for each user')
    parser.add_argument('--user', help='Select user running the vehicle', default="anirban",
                        choices=["anirban", "sugandh"])
    parser.add_argument('--x_min', help='x_min threshold', choices=[-6, -4])
    parser.add_argument('--y_min', help='y_min threshold', choices=[-4, -3])
    parser.add_argument('--x_max', help='x_max threshold', choices=[6, 4])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArg()
    mmwave_path = '/home/argha/Documents/driver-head-pose/'
    df = get_final_df("anirban", -6, -4, 6)
    print(df.shape)
    print(df.head(10))
    df.to_csv(mmwave_path + f"final_{args.user}_df.csv", index=False)
