# 🧠 Artificial (ANN) vs Convolutional (CNN)

## ☁️ Summary

The aim of this project is to determine the performance between an `Artificial neural network` (ANN) and `Convolutional neural network` (CNN) in classifying an image dataset. Although ANN is not preferred while working on images, it is good at recognizing patterns in any complex problems. Therefore, as a part of learning, this test was carried on and well it performed poorly on images with an accuracy of **~56%** where as CNN performed decent enough with an accuracy of **~77%**. We can say that convolutional neural network performs better than artificial neural networks in the classification of images.
> *The results can be improved by tuning hyperparameters.*

Access Jupyter Notebook: [Notebook](notebook.ipynb)

## 💻 Code Snippets

1. ANN Model
```python
ann_model = Sequential()

ann_model.add(Flatten(input_shape = (32,32,3)))
ann_model.add(Dense(3000, activation= 'relu'))
ann_model.add(Dense(1000, activation= 'relu'))
ann_model.add(Dense(10, activation= 'softmax'))
```

2. CNN Model
```python
cnn_model = Sequential()

cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model.add(MaxPooling2D((2, 2)))
    
cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
    
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))
```
