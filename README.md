# NeuroThreads 🧠👕

NeuroThreads is an advanced fashion image classification project that utilizes neural networks and deep learning techniques to accurately identify different types of clothing items. The project is built using TensorFlow and tf.keras, powerful libraries for machine learning and deep neural networks.

## Summary 📝

NeuroThreads is designed to classify fashion images into various categories such as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. It leverages the Fashion MNIST dataset, which consists of thousands of labeled fashion images for training and evaluation.

The neural network model employed in this project achieves high accuracy in classifying fashion items, enabling applications such as automated fashion tagging, recommendation systems, and inventory management in the fashion industry.

## Features ✨

- Neural network-based fashion image classification
- Utilizes TensorFlow and tf.keras libraries
- Achieves high accuracy in predicting fashion categories
- Interactive visualization of test results through plots and graphs

## Requirements 🛠️

The following third-party libraries are required to run NeuroThreads:

- [TensorFlow](https://www.tensorflow.org/) 2.6.0 or higher
- [NumPy](https://numpy.org/) 1.19.5 or higher
- [Matplotlib](https://matplotlib.org/) 3.4.3 or higher

You can install the dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
```

Make sure you have Python 3.7 or higher installed on your system.

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/Vpadia717/NeuroThreads.git
cd NeuroThreads
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Import the necessary libraries:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

2. Load the Fashion MNIST dataset and preprocess the data:

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Data preprocessing steps
train_images = train_images / 255.0
test_images = test_images / 255.0
```

3. Define the plotting functions for visualizing results:

```python
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                          100*np.max(predictions_array),
                                          class_names[true_label]),
                                          color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
```

4. Define and train the neural network model (Model Architecture):

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
```

5. Evaluate the model and visualize the results:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

# Plotting the first 25 test images, their predicted labels, and the true labels
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plot_image(i, predictions[i], test_labels, test_images)
plt.show()
```

## Test Cases 🧪

1. Testing inference on a single fashion image:

```python
# Select a random fashion image from the test dataset
image_index = np.random.randint(0, len(test_images))
test_image = test_images[image_index]
true_label = test_labels[image_index]

# Preprocess the image
input_image = (np.expand_dims(test_image, 0)) / 255.0

# Perform inference
predictions = probability_model.predict(input_image)
predicted_label = np.argmax(predictions[0])

# Display the result
print("True Label:", class_names[true_label])
print("Predicted Label:", class_names[predicted_label])
```

The above code snippet demonstrates a test case where inference is performed on a randomly selected fashion image. The true label and predicted label are displayed.

## Future Enhancements 🔮

The NeuroThreads project can be further enhanced in the following ways:

1. Integration with a live camera feed for real-time fashion classification.
2. Development of a user-friendly web or mobile application for fashion recognition.
3. Extension to multi-label classification to identify combinations of clothing items.

Contributions and suggestions for further improvements are

welcome!

## License 📜

This project is licensed under the [MIT License](LICENSE).
