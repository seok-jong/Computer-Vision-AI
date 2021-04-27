# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# Tensorflow가 활용할 GPU가 장착되어 있는지 확인해 봅니다.
tf.config.list_physical_devices('GPU')




def build_plainnet_block(input_layer,
                    num_cnn=2, 
                    channel=64,
                    block_num=1,
                    is_50=False
                   ):
    # 입력 레이어c
    x = input_layer
    
    
    # CNN 레이어
    if not is_50:
        for cnn_num in range(num_cnn):

            if (cnn_num==0):
                if (block_num != 0):
                    x = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
                x = keras.layers.Conv2D(
                    filters=channel,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv1'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_1")(x)
                x = keras.layers.Activation('relu')(x)

                x = keras.layers.Conv2D(
                    filters=channel,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv2'
                )(x)
#                 shortcut = keras.layers.Conv2D(
#                     filters=channel,
#                     kernel_size=(3,3),
#                     kernel_initializer='he_normal',
#                     padding='same',
#                     name=f'stage_{block_num+2}_{cnn_num+1}_conv2'
#                 )(input_layer)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_2")(x)
#                 shortcut=keras.layers.BatchNormalization(name=f"stage_{block_num+2}_shortcut{cnn_num+1}_2")(x)
#                 x = keras.layers.Add()([x, shortcut])
                x = keras.layers.Activation(activation='relu')(x)

            else:
                #shortcut=x
                x = keras.layers.Conv2D(
                    filters=channel,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv1'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_1")(x)
                x = keras.layers.Activation('relu')(x)

                x = keras.layers.Conv2D(
                    filters=channel,
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv2'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_2")(x)
                #x = keras.layers.Add()([x, shortcut])
                x = keras.layers.Activation('relu')(x)

                
    else:
        for cnn_num in range(num_cnn):
        
            if (cnn_num==0):
                if (block_num != 0):
                    x = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
                x = keras.layers.Conv2D(
                    filters=channel[0],
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv1'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_1")(x)
                x = keras.layers.Activation('relu')(x)

                x = keras.layers.Conv2D(
                    filters=channel[1],
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv2'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_2")(x)
                x = keras.layers.Activation('relu')(x)
                x = keras.layers.Conv2D(
                    filters=channel[2],
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv3'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_3")(x)
                
#                 shortcut = keras.layers.Conv2D(
#                     filters=channel[2],
#                     kernel_size=(1,1),
#                     kernel_initializer='he_normal',
#                     padding='same',
#                     name=f'stage_{block_num+2}_{cnn_num+1}_conv3'
#                 )(input_layer)
#                 shortcut=keras.layers.BatchNormalization(name=f"stage_{block_num+2}_shortcut{cnn_num+1}_2")(x)
                
                #x = keras.layers.Add()([x, shortcut])
                x = keras.layers.Activation(activation='relu')(x)

            else:
                #shortcut=x
                x = keras.layers.Conv2D(
                    filters=channel[0],
                    kernel_size=(1,1),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv1'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_1")(x)
                x = keras.layers.Activation('relu')(x)

                x = keras.layers.Conv2D(
                    filters=channel[1],
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv2'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_2")(x)
                x = keras.layers.Activation('relu')(x)
                x = keras.layers.Conv2D(
                    filters=channel[2],
                    kernel_size=(3,3),
                    kernel_initializer='he_normal',
                    padding='same',
                    name=f'stage_{block_num+2}_{cnn_num+1}_conv3'
                )(x)
                x = keras.layers.BatchNormalization(name=f"stage_{block_num+2}_bn{cnn_num+1}_3")(x)
                #x = keras.layers.Add()([x, shortcut])
                x = keras.layers.Activation(activation='relu')(x)

    # Max Pooling 레이어
#     x = keras.layers.MaxPooling2D(
#         pool_size=(2, 2),
#         strides=2,
#         name=f'block{block_num}_pooling'
#     )(x)

    return x




def build_plainnet(input_shape=(32,32,3),
            num_cnn_list=[3,4,6,3],
            channel_list=[64,128,256,512],
            num_classes=10,
            is_50=False,
            ):

    assert len(num_cnn_list) == len(channel_list) #모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.
    
    input_layer = keras.layers.Input(shape=input_shape)  # input layer를 만들어둡니다.
    output = input_layer
    
    output = output
    output = keras.layers.Conv2D(64, (7,7), kernel_initializer='he_normal', padding='same', strides=2,name="stage1_conv")(output)
    output = keras.layers.BatchNormalization(name="stage1_batchnomalization")(output)
    output = keras.layers.Activation('relu')(output)
    
    if is_50:
        channel_list=[[64,64,256],[128,128,512],[256,256,1024],[512,512,2048]]
        
    #config list들의 길이만큼 반복해서 블록을 생성합니다.
    for i, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
        if i ==0:
            output = keras.layers.MaxPooling2D((3, 3), strides=2, padding='same',name="stage2_0_maxpooling")(output)
        
        output = build_plainnet_block(
            output,
            num_cnn=num_cnn, 
            channel=channel,
            block_num=i,
            is_50=is_50
        )
        
    output = keras.layers.AveragePooling2D(padding="same")(output)
    output = keras.layers.Flatten(name='flatten')(output)
    output = keras.layers.Dense(10, activation='relu', name='fc1000')(output)
    #output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(output)
    
    model = keras.Model(
        inputs=input_layer, 
        outputs=output
    )
    return model




plainnet_34 = build_plainnet(input_shape=(32, 32,3), is_50=False)

plainnet_34.summary()





# dataset load

import urllib3
urllib3.disable_warnings()

#tfds.disable_progress_bar()   
# 이 주석을 풀면 데이터셋 다운로드과정의 프로그레스바가 나타나지 않습니다.

(ds_cd_train, ds_cd_test), ds_cd_info = tfds.load(
    'cats_vs_dogs',
    split=["train[:90%]","train[90%:]"],
    shuffle_files=True,
    as_supervised = True,
    with_info=True,
)


def normalize_and_resize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.image.resize(image, [32, 32])
    return tf.cast(image, tf.float32) / 255., label


def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=1
    )
    ds = ds.batch(batch_size)
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

BATCH_SIZE = 256
EPOCH = 10
ds_cd_train = apply_normalize_on_dataset(ds_cd_train, batch_size=BATCH_SIZE)
ds_cd_test = apply_normalize_on_dataset(ds_cd_test, batch_size=BATCH_SIZE)




plainnet_34.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.),
    metrics=['accuracy'],
)

history_plainnet_34 = plainnet_34.fit(
    ds_cd_train,
    steps_per_epoch=int(18610/BATCH_SIZE),
    validation_steps=int(2326/BATCH_SIZE),
    epochs=EPOCH,
    validation_data=ds_cd_test,
    verbose=1,
    use_multiprocessing=True,
)

resnet34.save('/home/ssac16/aiffel/going_deeper_CV/2_ResNet_Ablation_Study/plainnet34_w.h5')