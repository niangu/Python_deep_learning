#Keras是TensorFlow官方支持的，当机器上有可用的GPU时，代码会自动调用GPU进行并行计算
#优点
#模块化：模型的各个部分，如神经层，成本函数，优化器，初始化，激活函数，规范化，都是独立的模块，可以组合在一起创建模型
#极简主义：每个模块都保持简短和简单
#易扩展性：很容易添加新模块，适于做进一步的高级研究
#使用Python语言,模型用Python实现
#Keras核心数据结构是模型。模型是用来组织网络层的方式：Sequential模型，Model模型
'''
#假设数据以及加载
#Sequential模型的使用
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(outpot_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

#编译模型,同时指明损失函数和优化器
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#训练和评估模型
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
'''

#实现卷积神经网络（CNN）
#加载数据，模型构建，模型编译，模型训练，模型评估
#定义超参数以及加载数据
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

batch_size = 128
nb_classes = 10#分类数
nb_epoch = 12 #训练轮数

#输入图片的维度
img_rows, img_cols = 28, 28
#卷积滤镜的个数
nb_filters = 32
#最大池化， 池化核大小
pool_size = (2, 2)
#卷积核大小
kernel_size = (3, 3)

#(X_train, y_train), (X_test, y_test) = mnist.load_data('data2/')
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

if K.image_dim_ordering() == 'th':
    #使用Theano的顺序：（conv_dim1, channels, conv_dim2, conv_dim3）
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    #使用TensorFlow的顺序：（conv_dim1, conv_dim2, conv_dim3, channels）
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#将类向量转换为二进制类矩阵
Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)

#下面构建模型，2个卷积层，1个池化层，2个全连接层
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size[0],
                 kernel_size[1], activation='relu', border_mode='valid',
                 input_shape=input_shape))

model.add(Conv2D(nb_filters, kernel_size[0],
                 kernel_size[1], activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#模型的加载及保存：
#Keras的model.save(filepath)和keras,models.load_model(filepath)方法可以将Keras模型和权重保存在一个HDF5文件中，这里包括模型的结构，权重，
# 训练的配置（损失函数，优化器），优化程序的状态(保证可以从上次中断的地方继续正确的训练模型)等
from keras.models import load_model

model.save('my_model.h5') #creates a HDF5 file 'my_model.h5'
del model #deletes the existing model

#returns a compiled model
#identical to the precious one
model = load_model('my_model.h5')

#只保存模型的体系结构，而不是模型的权重或训练配置，则执行以下操作：
#save as JSON
json_string = model.to_json()
#save as YAML
yaml_string = model.to_yaml()
#生成的JSON/YAML文件是人类可读的，并且可以根据需要进行手动编辑
#然后，您可以根据以下数据构建新的模型
#model reconstruction from JSON
from keras.models import model_from_json
model = model_from_json(json_string)

#model reconstruction from TAML:
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
#仅保存/加载模型的权重
model.save_weights('my_model_weights.h5')
#假设你有用于实例化模型的代码，然后可以将保存的权重加载到具有相同架构的模型中：
model.load_weights('my_model_weights.h5')
#如果需要将权重加载到不同的体系结构中（共有一些层），例如进行微调或转移学习，则可以按层名称加载它们
model.load_weights('my_model_weights.h5', by_name=True)
