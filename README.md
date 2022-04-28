# Deep Learning for Web Attack Detection 
Detecting Web Attacks ( *Sqli* and *XSS* ) from URLs with deep learning models
The detection accuracy of this project is about 98.3%.


## Details
All data in this project are from public datasets. Here we provide a good website for [Sercrity Datsets](http://www.secrepo.com/)

Folder **''Code''** : python files for this work
- DataProcess: python file for data process.
- Model: python file for CNN model.

Folder **''Data''** : data for detection.  
- Unique-data: row data for this project, the normal and abnormal requests are in separate files.
- Train&Test: mixed normal and abnormal requests, and generated train and test dataset (row data). 
- Model-Word2vec: Trained Word2vec model.
- Replace-data: files for keywords generated, train and test files repalced with keywords.
- Encode-data: train and test files encoded by Word2vec model.
- Predict: predicted labels for test data. 

Folder **''Model_Save''**: trained model.

## Dependent Libraries
- python 3.63-64-bit 
- numpy 1.16.4
- tensorflow 1.8.0
- keras 2.2.2
- sklearn 0.19.1
- gensim 3.5.0

## Useage
- Python files need to run in the root path of the project. 
- run data_process.py to generate encoded data firstly.
- run keras_v1_One_label.py to train/evaluate the model.

Note that the source data need to be splited by words and punctuations if u try to use you own data. 

*The provided features and deep learning models in this project are very simple and you can add or create your own features and models based on this project.* &nbsp; : )



## Cite this work
This project is a part job of our publised works in IEEE Transactions on Industrial Informatics. You can cite this work in your researches.

"A Novel Web Attack Detection System for Internet of Things via Ensemble Classification," in IEEE Transactions on Industrial Informatics, vol. 17, no. 8, pp. 5810-5818, Aug. 2021, doi: 10.1109/TII.2020.3038761. 
[Paper Link](https://ieeexplore.ieee.org/document/9261992)



