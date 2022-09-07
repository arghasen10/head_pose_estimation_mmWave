import datetime
from models import read_data,Model,rf_model,cnn_model,vgg16_model

runtime=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

#exp1-merged
train_pattern="../*_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test=read_data(train_pattern=train_pattern,class_count=1000)
#rf+vgg16+cnn_model
for md_class in [rf_model,vgg16_model,cnn_model]:
    model=Model(X_train, X_test, y_train, y_test,md_class)
    model.train()
    model.test('merged'+runtime)

#exp2-static
train_pattern="../static_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test=read_data(train_pattern=train_pattern,class_count=600)
#rf+vgg16+cnn_model
for md_class in [rf_model,vgg16_model,cnn_model]:
    model=Model(X_train, X_test, y_train, y_test,md_class)
    model.train()
    model.test('static'+runtime)

#exp3-driving
train_pattern="../driving_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test=read_data(train_pattern=train_pattern,class_count=600)
#rf+vgg16+cnn_model
for md_class in [rf_model,vgg16_model,cnn_model]:
    model=Model(X_train, X_test, y_train, y_test,md_class)
    model.train()
    model.test('driving'+runtime)

