import datetime
from models import read_mydata, Model, rf_model, cnn_model, vgg16_model

runtime = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
saved_paths = ['{}', './saved_weights/' + runtime + '_vgg16_{}.h5', './saved_weights/' + runtime + '_cnn_{}.h5']
# exp1-merged
train_pattern = "../*_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, class_count=1000)
# rf+vgg16+cnn_model
for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
    model = Model(X_train, X_test, y_train, y_test, md_class)
    model.train(sp.format('merged'))
    model.test('merged' + runtime)

# exp2-static
train_pattern = "../static_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, class_count=600)
# rf+vgg16+cnn_model
for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
    model = Model(X_train, X_test, y_train, y_test, md_class)
    model.train(sp.format('static'))
    model.test('static' + runtime)

# exp3-driving
train_pattern = "../driving_dataset/processed/denoised/*"
X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, class_count=600)
# rf+vgg16+cnn_model
for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
    model = Model(X_train, X_test, y_train, y_test, md_class)
    model.train(sp.format('driving'))
    model.test('driving' + runtime)

# exp4-leave-one-out
# static
for user in ['argha', 'anirban', 'bishakh', 'aritra']:
    train_pattern = f"../static_dataset/processed/denoised/*!{user}"
    test_pattern = f"../static_dataset/processed/denoised/*_{user}*"

    X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, test_pattern=test_pattern,
                                                   class_count=200)
    # rf+vgg16+cnn_model
    for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
        model = Model(X_train, X_test, y_train, y_test, md_class)
        model.train(sp.format(f'static_{user}_loO'))
        model.test(f'static_{user}_loO' + runtime)

# Driving
for user in ['bishakh3', 'anirban1', 'bishakh2', 'sugandh3', 'anirban2', 'sugandh2', 'bishakh1', 'sugandh1']:
    train_pattern = f"../driving_dataset/processed/denoised/*!{user}"
    test_pattern = f"../driving_dataset/processed/denoised/*_{user}*"

    X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, test_pattern=test_pattern,
                                                   class_count=200)
    # rf+vgg16+cnn_model
    for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
        model = Model(X_train, X_test, y_train, y_test, md_class)
        model.train(sp.format(f'driving_{user}_loO'))
        model.test(f'driving_{user}_loO' + runtime)

# exp5-personalized
# static
for user in ['argha', 'anirban', 'bishakh', 'aritra']:
    train_pattern = f"../static_dataset/processed/denoised/*_{user}*"

    X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, class_count=200)
    # rf+vgg16+cnn_model
    for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
        model = Model(X_train, X_test, y_train, y_test, md_class)
        model.train(sp.format(f'static_{user}_per'))
        model.test(f'static_{user}_per' + runtime)

# Driving
for user in ['bishakh3', 'anirban1', 'bishakh2', 'sugandh3', 'anirban2', 'sugandh2', 'bishakh1', 'sugandh1']:
    train_pattern = f"../driving_dataset/processed/denoised/*_{user}*"

    X_train, X_test, y_train, y_test = read_mydata(train_pattern=train_pattern, class_count=200)
    # rf+vgg16+cnn_model
    for md_class, sp in zip([rf_model, vgg16_model, cnn_model], saved_paths):
        model = Model(X_train, X_test, y_train, y_test, md_class)
        model.train(sp.format(f'driving_{user}_per'))
        model.test(f'driving_{user}_per' + runtime)
