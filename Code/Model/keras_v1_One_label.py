
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')
import keras 
import pandas as pd
import numpy as np




def cnn_model(data_file_name,label_file_name,model_save_path):
    # input
    input_A=keras.layers.Input(shape=(80,48,1))
    # CNN part
    layers=keras.layers.Conv2D(128,(1,3),padding='same',activation='relu')(input_A)
    
    layers=keras.layers.Conv2D(128,(1,3),padding='same',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(1,2),padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(1,3),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(1,2),padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(1,3),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(1,2),padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(1,3),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(1,2),padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(1,3),padding='VALID',activation='relu')(layers)
    # layers=keras.layers.MaxPooling2D(padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(3,1),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(2,1),padding='SAME')(layers)

    layers=keras.layers.Conv2D(128,(3,1),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(2,1),padding='SAME')(layers)

    layers=keras.layers.Conv2D(128,(3,1),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(2,1),padding='SAME')(layers)
    
    layers=keras.layers.Conv2D(128,(3,1),padding='SAME',activation='relu')(layers)
    layers=keras.layers.MaxPooling2D(pool_size=(2,1),padding='SAME')(layers)

    # layers=keras.layers.Conv2D(12,(5,48),padding='valid')(input_A)
    # layers=keras.layers.Conv2D(12,(7,48),padding='valid')(input_A)
    # layers=keras.layers.Conv2D(12,(11,48),padding='valid')(input_A)
    
    layers=keras.layers.Flatten()(layers)
    layes=keras.layers.Dropout(0.5)(layers)
    # FC part
    layers=keras.layers.Dense(128,activation='relu')(layers)
    layes=keras.layers.Dropout(0.5)(layers)

    # layers=keras.layers.Dense(64)(layers)
    layers=keras.layers.Dense(40,activation='relu')(layers)
    # layers=keras.layers.Dense(4)(layers)
    
    # layers_end=keras.layers.Dense(1,activation='relu')(layers)

    # -----------------change 
    layers_end=keras.layers.Dense(1,activation='sigmoid')(layers)

    # model compile
    model=keras.Model(inputs=input_A,outputs=layers_end)
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    model.summary()


    # load data & train
    file_data=data_reshape(data_file_name)
    # label_data=pd.read_csv(label_file_name,header=None).values
    label_data=np.loadtxt(label_file_name,delimiter=',')
    print(np.shape(label_data))
    model.fit(file_data,label_data,batch_size=64,epochs=8)
    model.save(model_save_path)
    print('-------------- Trianing process finished')
    #  evaluate 
    # predict 


def data_reshape(file_name):
    file_data=pd.read_csv(file_name,header=None).values
    print(np.shape(file_data))
    num_data=np.shape(file_data)[0]
    reshape_data=file_data.reshape((num_data,80,48,1))
    return reshape_data

def model_predict(data_file_name,label_file_name,model_path):
    file_data=data_reshape(data_file_name)
    label_data=pd.read_csv(label_file_name,header=None).values
    Model=keras.models.load_model(model_path)
    score=Model.evaluate(file_data,label_data)
    print('The loss and accuracy are calculated: ')
    print(score)
    print('-------------- Evaluating process finished')

    # acc=keras.metrics.categorical_accuracy(pre_data,label_data)
    # print('acc:',acc)



if __name__ == "__main__":
    folder=''
    # ---------------------------- Train CNN model 
    data_file_name=folder+'Data/ALL-New-V2/Encode-data/encode_train.csv'
    label_file_name=folder+'Data/ALL-New-V2/Train&Test/label_train.csv'
    model_save_path=folder+'Model_Save/keras_v1_One_label.h5'
    cnn_model(data_file_name,label_file_name,model_save_path)

    # ---------------------------- Evaluate the model 
    data_file_name=folder+'Data/ALL-New-V2/Encode-data/encode_test.csv'
    label_file_name=folder+'Data/ALL-New-V2/Train&Test/label_test.csv'
    model_save_path='Model_Save/keras_v1_One_label.h5'
    model_predict(data_file_name,label_file_name,model_save_path)
    

    # ---------------------------- predict 
    data_file_name=folder+'Data/ALL-New-V2/Encode-data/encode_test.csv'
    file_data=data_reshape(data_file_name)
    model_save_path='Model_Save/keras_v1_One_label.h5'
    Model=keras.models.load_model(model_save_path)
    file_name='Data/ALL-New-V2/Predict/predict_one_label.csv'
    pre_data=Model.predict(file_data)
    np.savetxt(file_name,pre_data,fmt='%2f')




