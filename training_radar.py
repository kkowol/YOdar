import numpy as np
import time

from keras.layers import *
from keras.models import *
from keras.optimizers import *

from keras.callbacks import *
import keras.backend as K
from sklearn.metrics import roc_curve, auc
import config as cfg
from contextlib import redirect_stdout # for saving summary

## chose the right GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= cfg.CUDA_VISIBLE_DEVICES

def my_acc( y_true, y_pred ):
    t = K.variable(0.5)
    g = K.greater( y_pred, t )
    g = K.cast( g, "float32" )
    r = K.equal( y_true, g )
    r = K.flatten(r)
    return K.mean(r)

def my_loss(y_true, y_pred):
    return - 4.0*K.mean( y_true * K.log( y_pred + 1e-10 ) ) - K.mean( (1-y_true)* K.log( (1-y_pred) + 1e-10 ) ) # higher weighting of true values

def plot_roc_curve(Y, probs, roc_path):
  fpr, tpr, _ = roc_curve(Y, probs)
  roc_auc = auc(fpr, tpr)
  print("auc", roc_auc)
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve of meta classification performance')
  plt.legend(loc="lower right")
  roc_dir  = os.path.dirname( roc_path )
  if not os.path.exists( roc_dir ):
    os.makedirs( roc_dir )
  plt.savefig(roc_path)
  print("roc curve saved to " + roc_path)
  plt.close()
  return roc_auc

def calculations(pr, y_data):
    gt = y_data.reshape((pr.shape[0], pr.shape[1],))
    pr = pr.reshape(pr.shape[0], pr.shape[1])
    TP = len(['j:{}, i:{}'.format(j, i) for j in range(len(pr)) for i in range(len(gt[0])) if
              gt[j][i] == 1 and pr[j][i] == 1])
    FP = len(['j:{}, i:{}'.format(j, i) for j in range(len(pr)) for i in range(len(gt[0])) if
              gt[j][i] == 0 and pr[j][i] == 0])
    FP = len(['j:{}, i:{}'.format(j, i) for j in range(len(pr)) for i in range(len(gt[0])) if
              gt[j][i] == 0 and pr[j][i] == 1])
    FN = len(['j:{}, i:{}'.format(j, i) for j in range(len(pr)) for i in range(len(gt[0])) if
              gt[j][i] == 1 and pr[j][i] == 0])
    IoU = round(((TP) / (FP + TP + FN))*100,2)
    precision = round(((TP) / (TP + FP)) * 100, 2)
    recall = round(((TP) / (TP + FN)) * 100,2)
    return IoU, precision, recall


def start_training(epochs, weightdecay, batch_size, name_affix, path,
                   x_train, y_train, x_val, y_val, x_test, y_test):

    train_size,slices, timesteps, channels = x_train.shape
    val_size = x_val.shape[0]
    test_size = x_test.shape[0]

    input_shape = (slices,timesteps,channels)
    output_shape = (slices,1,1)

    base_filters = 32
    kernel_size = (5,3)
    strides = (2,1)
    dropout = 0.25
    f = base_filters
    inp = Input(shape=input_shape)
    kernel_initializer = 'glorot_uniform'

    # conv block 1
    y1 = Conv2D(filters=f, kernel_size=kernel_size, padding='same', strides=strides,
                kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(inp)
    y1 = BatchNormalization()(y1)
    y = LeakyReLU(0.1)(y1)

    # conv block 2
    y2 = Conv2D(filters=f * 2, kernel_size=kernel_size, padding='same', strides=strides,
                kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y2 = BatchNormalization()(y2)
    y = LeakyReLU(0.1)(y2)

    # conv block 3
    y3 = Conv2D(filters=f * 4, kernel_size=kernel_size, padding='same', strides=strides,
                kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y3 = BatchNormalization()(y3)
    y = LeakyReLU(0.1)(y3)

    # deconv block 1
    y = Conv2DTranspose(filters=f * 2, kernel_size=kernel_size, padding='same', strides=strides,
                        kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    # add output from 2nd conv
    y = Add()([y, y2])
    y = LeakyReLU(0.1)(y)
    

    # deconv block 2
    y = Conv2DTranspose(filters=f, kernel_size=kernel_size, padding='same', strides=strides,
                        kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    # add output from 1st conv
    y = Add()([y, y1])
    y = LeakyReLU(0.1)(y)
    
    # deconv block 3
    y = Conv2DTranspose(filters=f, kernel_size=kernel_size, padding='same', strides=strides,
                        kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)

    # concatenate input
    y = Concatenate()([y, inp])
    y = LeakyReLU(0.1)(y)

    # final conv for downsizing to desired number of classes
    y = Conv2D(filters=12, kernel_size=kernel_size, padding='same', strides=(1,1),
                        kernel_regularizer=regularizers.l2(weightdecay), kernel_initializer=kernel_initializer)(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(0.1)(y)

    # softmax in the filter dimension
    y = Flatten()(y)
    y = Dense(units=slices, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(weightdecay))(y)

    y = Reshape((slices,1,1))(y)
    y  = Activation('sigmoid')(y)

    # -------------------------------------------------------------------------------
    # define model
    model = Model(inputs=inp, outputs=y)

    model.summary()
    if not os.path.exists(path + "/net"):
        os.makedirs(path + "/net")
    with open(path + '/net/modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    x_train = x_train.reshape((train_size,)+input_shape)
    y_train = y_train.reshape((train_size,)+output_shape)
    x_val = x_val.reshape((val_size,)+input_shape)
    y_val = y_val.reshape((val_size,)+output_shape)

    mean_x_train = np.zeros((channels,))
    std_x_train = np.zeros((channels,))
    for i in range(channels):
        mean_x_train[i] = np.mean(x_train[:,:,:,i].flatten())
        std_x_train[i] = np.std(x_train[:, :, :, i].flatten())

    if not os.path.exists('./norm'):
        os.makedirs('./norm')
    np.save('./norm/channel_mean.npy', mean_x_train)
    np.save('./norm/channel_std.npy', std_x_train)

    for i in range(channels):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean_x_train[i] ) / std_x_train[i]
        x_val[:, :, :, i] = (x_val[:,:,:,i] - mean_x_train[i] ) / std_x_train[i]


    # train the model
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=my_loss, metrics=[my_acc])
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              #callbacks=[early_stopping],
              shuffle=True)

    adam = Adam(lr=lr/10, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=my_loss, metrics=[my_acc])
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=int(epochs/2),
              validation_data=(x_val, y_val),
              #callbacks=[early_stopping],
              shuffle=True)

    adam = Adam(lr=lr/100, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=my_loss, metrics=[my_acc])
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=int(epochs/2),
              validation_data=(x_val, y_val),
              #callbacks=[early_stopping],
              shuffle=True)

    ### save/load model
    model.save(path + '/net/fcn_net.h5')

    ### Validation
    score = model.evaluate(x_val, y_val, batch_size=batch_size)

    pr = model.predict(x_val, batch_size=batch_size) > 0.5
    IoU_val, precision_val, recall_val = calculations(pr, y_val)

    print("Validation IoU:", IoU_val)
    print('Precision: ', precision_val, '%, Recall: ', recall_val, '%')
    print("Validation performance: ", round(score[1]*100, 2), '%')
    print('Calculation for:', name_affix[0])

    ### testing with unknown images
    x_test = x_test.reshape((test_size,)+input_shape)
    y_test = y_test.reshape((test_size,)+output_shape)

    for i in range(channels):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean_x_train[i]) / std_x_train[i]

    score_unkown = model.evaluate(x_test, y_test, batch_size=batch_size)

    pr = model.predict(x_test, batch_size=batch_size) > 0.5 # prediction
    pr = pr.reshape(pr.shape[0], pr.shape[1])
    IoU_test, precision_test, recall_test = calculations(pr, y_test)

    print("Test IoU:", IoU_test )
    print('Precision: ', precision_test, '%, Recall: ', recall_test, '%')
    print("Test performance: ", round(score_unkown[1] * 100, 2), '%')
    print('Calculation for:', name_affix[0])

    # write Informations in Textfile
    with open( path + "/net/Output.txt", "w") as text_file:
        print('Data:', name_affix[0], name_affix[1] , file = text_file)
        print("Epochs: {}, Batchsize: {}, Weightdecay: {}".format(epochs,batch_size, weightdecay), file=text_file)
        print('Training/Validation/Test Samples: {} / {} / {}'.format(train_size, val_size, test_size), file=text_file)
        print('Test IoU: {}%'.format(IoU_test), file=text_file)
        print('Precision: {}%'.format(precision_test), ' Recall: {}%'.format(recall_test), file=text_file)
        print("Test performance: {}".format(round(score_unkown[1]*100, 2)), '%\n', file=text_file)


if __name__ == '__main__':
    start = time.time()

    path = cfg.path
    name_affix = cfg.name_affix
    batch_size = cfg.batch_size
    weightdecay = cfg.weightdecay
    epochs = cfg.epochs
    lr = cfg.lr

    x_train = cfg.x_train
    y_train = cfg.y_train
    x_val = cfg.x_val
    y_val = cfg.y_val
    x_test = cfg.x_test
    y_test = cfg.y_test

    start_training(epochs, weightdecay, batch_size, name_affix, path,
                   x_train, y_train, x_val, y_val, x_test, y_test)

    end = time.time()
    print('Time: %ds = %dmin' % (end - start, (end - start) / 60))