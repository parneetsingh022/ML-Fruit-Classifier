# ML-Fruit-Classifier

This notebook demonstrates how to build and train a convolutional neural network (CNN) to classify images of fruits. The dataset contains 732 images of bananas, mangoes and watermelons, divided into 3 classes.

<img src="https://github.com/parneetsingh022/ML-Fruit-Classifier/assets/99793808/bb7db8ed-0709-4739-990e-068645eb5dd7" width="500px"><br>

The notebook uses TensorFlow 2.7 and Keras to create the model, which consists of the following layers:
- Data augmentation: A sequential layer that applies random transformations to the input images, such as flipping, rotating and zooming, to increase the diversity and robustness of the training data.
```python
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)
```
- Rescaling: A layer that rescales the pixel values of the images from [0, 255] to [0, 1], which helps the model converge faster.
```python
tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
```
- Conv2D: A layer that performs 2D convolution, which is a mathematical operation that extracts features from the input images using a set of filters.
- MaxPooling2D: A layer that reduces the spatial dimensions of the feature maps by taking the maximum value in each window.
- Dropout: A layer that randomly drops out some units from the previous layer, which helps prevent overfitting and improve generalization.
- Flatten: A layer that flattens the feature maps into a 1D vector, which can be fed into a dense layer.
- Dense: A layer that performs a linear transformation on the input vector, followed by an activation function. The last dense layer has 3 units, corresponding to the number of classes, and uses a softmax activation to output the class probabilities.

The model is compiled with an Adam optimizer, a sparse categorical crossentropy loss function, and an accuracy metric. The model is trained for 200 epochs on the augmented and shuffled image dataset, using a batch size of 32. The model achieves an accuracy of 0.9385 on the training set.

The notebook also shows how to use the model to make predictions on new images, using the class_names list to map the output indices to the fruit names. The notebook displays the input image and the predicted class name for each test image.

The image dataset is not provided because of its huge size. But this model could be implimented on any images (need not to be fruits). One have to write the correct location of the folder containing all dataset files which are divided further into sub-folders according to their labels.

```python
image_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR,'images'),
    batch_size=32,
    image_size=(img_width,img_height),
    shuffle=True,
    seed=123,
    color_mode='rgb'
)
```
