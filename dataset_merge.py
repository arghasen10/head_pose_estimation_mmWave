import pandas as pd
import glob
import numpy as np
import json

global_fps = 2
name = 'anirban_drive20'


def get_label(sort_dic):
    total_count = sum(sort_dic.values())
    label_val = ""
    for label, count in sort_dic.items():
        if count == total_count:
            return label  # For single activity in FPS (looking forward, yawning, talking)
        elif "looking left" == label:
            return label
        elif "looking right" == label:
            return label
        elif "looking up" == label:
            return label
    if "yawning" in sort_dic:
        return "yawning"
    elif "talking" in sort_dic:
        return "talking"
    else:
        return "looking forward"


def reformat_milli(df):
    df['datetime'] = [ts + f'_{i}' for ts, e in df.groupby('datetime').count().iloc[:, 0].to_dict().items() for i in
                      range(e)]
    return df


def read_image(filename):
    image_df = pd.read_csv(filename)
    image_df = image_df[['datetime', 'x', 'y', 'z', 'mar', 'activity']]
    return image_df


def process_image(image_df):
    g = image_df.groupby('datetime')
    ds_list = []
    for name, gr in g:
        stride = int(gr.shape[0] / global_fps)
        vals = np.array([gr.iloc[i * stride:i * stride + stride].drop(columns=['datetime', 'activity']).mean().values \
                         for i in range(global_fps)])
        major_act = np.array(
            [get_label(dict(sorted(gr.iloc[i * stride:i * stride + stride]['activity'].value_counts().to_dict().items(),
                                   key=lambda e: e[1], reverse=True))) for i in range(global_fps)])
        ds_gr = pd.DataFrame({'datetime': [name] * global_fps,
                              'x': vals[:, 0], 'y': vals[:, 1], 'z': vals[:, 2], 'mar': vals[:, 3],
                              'activity': major_act
                              })
        ds_list.append(ds_gr)
    image_df = pd.concat(ds_list, ignore_index=True)
    return image_df.dropna()


def process_mmWave(filename):
    data = [json.loads(val) for val in open(filename, "r")]
    mmwave_df = pd.DataFrame()
    for d in data:
        mmwave_df = mmwave_df.append(d['answer'], ignore_index=True)

    mmwave_df['datetime'] = mmwave_df['timenow'].apply(lambda e: '2022-09-03 ' + ':'.join(e.split('_')))
    mmwave_df = mmwave_df[['datetime', 'rp_y', 'noiserp_y', 'doppz']]
    return mmwave_df.dropna()


# file paths
base_path = '/home/argha/Documents/github/head_pose_estimation_mmWave/driving_dataset/dataset/'
mmwave_path = f'{base_path}/{name}**dataset.txt'
image_path = f'{base_path}/{name}**.csv'
mmwave_data = pd.concat([reformat_milli(process_mmWave(f)) for f in glob.glob(mmwave_path)])
image_df = reformat_milli(process_image(pd.concat([read_image(f) for f in glob.glob(image_path)])))
image_df.set_index('datetime', inplace=True)
mmwave_data.set_index('datetime', inplace=True)
processed = pd.concat([image_df, mmwave_data], join='inner', axis=1).reset_index()
processed.to_csv(base_path + f"/final_{name}_df.csv", index=False)
