from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import np_utils

# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(3, None, None))
#
# base_model = Model(input=resnet.input, output=resnet.layers[-2].output)
#
# top_model = Sequential()
# # top_model.add(base_model)
# top_model.add(Flatten(input_shape=(2048, 7, 7)))
# top_model.add(Dense(2, activation='softmax'))
#
# model = Sequential()
# model.add(base_model)
# model.add(Flatten()(base_model.output))
# model.add(Dense(2, activation='softmax'))
# # model.add(top_model)
# print np.shape(model.layers)
train_root = '/home/ty/data/saliency/MSRA5000/train'
validate_root = '/home/ty/data/saliency/MSRA5000/val'
image_size = (200, 200)
train_samples = 4500 + 4500*5
validate_samples = 500 + 500*5

print train_samples, validate_samples
resnet_mean = [103.939, 116.779, 123.68]

#using resnet pop ave_pooling to extract bottleneck feature
# feature shape [2048, 7, 7]
def extract_bottleneck_feature():
    datagen = ImageDataGenerator(
        rescale=1.,
        featurewise_center=True,
    )

    datagen.mean = np.array(resnet_mean, dtype=np.float32).reshape(3, 1, 1)

    train_generator = datagen.flow_from_directory(
        train_root,
        target_size=image_size,
        batch_size=24,
        class_mode='binary',
        shuffle=False
    )

    validate_generator = datagen.flow_from_directory(
        validate_root,
        target_size=image_size,
        batch_size=24,
        class_mode='binary',
        shuffle=False
    )

    base_model = ResNet50(include_top=False, weights='imagenet')
    feature_model = Model(input=base_model.input, output=base_model.layers[-2].output)

    bottleneck_train_feature = feature_model.predict_generator(train_generator, train_samples)
    np.save(open('bottleneck_train_feature.npy', 'w'), bottleneck_train_feature)

    bottleneck_validate_feature = feature_model.predict_generator(validate_generator, validate_samples)
    np.save(open('bottleneck_validate_feature.npy', 'w'), bottleneck_validate_feature)


# using tensorflow can complete bottleneck training, theano fail because trouble in cuda or cudnn

train_data = np.load(open('bottleneck_train_feature.npy'))
validate_data = np.load(open('bottleneck_validate_feature.npy'))

print train_data.shape

train_data = train_data.transpose(0, 2, 3, 1)
validate_data = validate_data.transpose(0, 2, 3, 1)
print train_data.shape

datagen = ImageDataGenerator(
    rescale=1.,
    featurewise_center=True,
)
datagen.mean = np.array(resnet_mean, dtype=np.float32).reshape(3, 1, 1)
train_generator = datagen.flow_from_directory(
    train_root,
    target_size=image_size,
    batch_size=24,
    class_mode='binary',
    shuffle=False
)

validate_generator = datagen.flow_from_directory(
    validate_root,
    target_size=image_size,
    batch_size=24,
    class_mode='binary',
    shuffle=False
)

train_labels = np_utils.to_categorical(train_generator.classes, 2)
validate_labels = np_utils.to_categorical(validate_generator.classes, 2)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, nb_epoch=30, batch_size=32, validation_data=(validate_data, validate_labels))
model.save_weights('bottleneck_fc_model.h5')