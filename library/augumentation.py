import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
from .helper import load_dataset

def scale(doppz,Max=5293,Min=1375):
    doppz_scaled=(doppz-Min)/(Max-Min)
    return doppz_scaled
    
def StackFrames(doppz,labels,frame_stack=4):
    max_index=doppz.shape[0]-frame_stack
    stacked_doppz=np.array([doppz[i:i+frame_stack] for i in range(max_index)]).transpose(0,2,3,1)
    new_labels=np.array([labels[i+frame_stack-1] for i in range(max_index)])
    return stacked_doppz,new_labels

def get_aug_pipe(frame_stack=4):
    Input=tf.keras.layers.Input(shape=(128,64,frame_stack))
    net=tf.keras.layers.Cropping2D(((0, 0), (0, 24)),name='Crop')(Input)
    net=tf.keras.layers.Resizing(height=48,width=48,name='Resize_48x48')(net)
    net=tf.keras.layers.RandomTranslation(height_factor=(0.0,0.0),
                                          width_factor=(0.0,0.8),fill_mode='wrap',name='R_shift')(net)
    pipe=tf.keras.Model(inputs=[Input],outputs=[net],name='Aug_pipe')
    return pipe
    
def run_aug_once(doppz_scaled_stacked,pipe):
    return pipe(doppz_scaled_stacked,training=True).numpy()

#Data Augumentation class
class Dataset:
    def __init__(self,loc="../static_dataset/processed/denoised/*",class_count=600,frame_stack=4,dop_min=1375,dop_max=5293):
        #Temp vals
        print(f"loading dataset from {loc}")
        df=load_dataset(loc)
        doppz=np.array(df['doppz'].values.tolist())
        label=df['activity'].values
        doppz_scaled_stacked,new_labels=StackFrames(scale(doppz,dop_max,dop_min),label,frame_stack)
        
        #class members
        self.class_count=class_count
        self.do_num_aug=np.ceil(self.class_count/df['activity'].value_counts()).to_dict()
        self.pipe=get_aug_pipe(frame_stack=frame_stack)
        self.data,self.label=self.process(doppz_scaled_stacked,new_labels)

    def augument(self,stacked_doppz_sub_arr,num_aug=None):
        total_arr=np.concatenate([run_aug_once(stacked_doppz_sub_arr,self.pipe) for _ in range(num_aug)],axis=0)
        final_indices=np.random.choice(np.arange(0,total_arr.shape[0]),size=self.class_count,replace=False)
        return total_arr[final_indices]
    
    def process(self,doppz_scaled_stacked,new_labels):
        data=[]
        lbl=[]
        for activ, num_aug in self.do_num_aug.items():
            print(f"on activity {activ} -> augument for {num_aug} times")
            data.append(self.augument(doppz_scaled_stacked[new_labels==activ],int(num_aug)))
            lbl.extend([activ]*self.class_count)
        data=np.concatenate(data,axis=0)
        lbl=np.array(lbl)
        return data,lbl