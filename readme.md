# Basic Skin Tone NN: A Simple Neural Network for Skin Tone Classification

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Made with NumPy](https://img.shields.io/badge/Made%20with-NumPy-blue.svg)](https://numpy.org/)

A simple, lightweight neural network built **entirely from scratch** with NumPy to classify skin tones from images. This project serves as a clear and concise example of the fundamentals behind building and training a neural network for a computer vision task without relying on high-level libraries like TensorFlow or PyTorch.

---

### 🧠 How It Works

The project operates in a few key stages:

1.  **Image Preprocessing**: All images are resized to a uniform `128x128` pixels and normalized. This ensures consistency and helps the model train effectively.
2.  **Dataset Loading**: The script loads images from the `dataset` directory, where each subdirectory is named after a skin tone class.
3.  **Model Architecture**: The core of the project is a `SimpleNN` class that defines a neural network with a single hidden layer. It uses the **Sigmoid** activation function for the hidden layer and **Softmax** for the output layer to produce classification probabilities.
4.  **Training**: The model is trained using batch gradient descent. The dataset is split into 70% for training and 30% for testing.
5.  **Prediction**: After training, the model can predict the skin tone of new images placed in the `unknown_images` folder and saves the predictions to `predictions_log.txt`.

---

### ✨ Features

| Feature                | Description                                        |
| :--------------------- | :------------------------------------------------- |
| **Neural Network** | Built **from scratch** using only NumPy.           |
| **Frameworks** | NumPy, OpenCV                                      |
| **Architecture** | Simple Neural Network (1 hidden layer)             |
| **Activation Functions** | Sigmoid, Softmax                                   |
| **Loss Function** | Cross-Entropy                                      |
| **Optimizer** | Batch Gradient Descent                             |
| **Image Size** | 128x128                                            |
| **Default Classes** | Light, Very Light, Medium, Deep, Very Deep         |

---

### 🗂️ Dataset & Customization

This project is designed to be flexible. You can use the recommended dataset or your own custom dataset.

#### Dataset Structure

The dataset must be organized into subdirectories within the `dataset/` folder. The name of each subdirectory serves as the class label for all the images within it.

```
dataset/
├── Class_1/
│   ├── image1.jpg
│   └── image2.png
├── Class_2/
│   ├── image3.jpg
│   └── ...
└── Class_3/
    └── ...
```

#### Recommended Dataset

For excellent results, the **[Skin Tone Classification Dataset on Kaggle](https://www.kaggle.com/datasets/usamarana/skin-tone-classification-dataset?resource=download)** is recommended. With this dataset, the model can achieve an **accuracy of approximately 90%**.

#### Using Your Own Classes

You can easily train the model on your own classes. To do so, you just need to modify the `CLASSES` list in the `main.ipynb` notebook.

For example, to use classes like 'Type A', 'Type B', and 'Type C', you would change this line:

```python
# In main.ipynb, modify this line:
CLASSES = ["Type A", "Type B", "Type C"]
```

Remember to organize your `dataset` directory to match these new class names.

---

### 🚀 Getting Started

#### **Prerequisites**

Make sure you have Python 3 and Jupyter Notebook installed.

#### **Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Basic-Skin-Tone-NN.git](https://github.com/your-username/Basic-Skin-Tone-NN.git)
    cd Basic-Skin-Tone-NN
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy opencv-python jupyter
    ```

#### **Usage**

1.  Prepare your `dataset` folder according to the structure mentioned above.
2.  Place any images you want to predict on in the `unknown_images` folder.
3.  Run the Jupyter Notebook to train the model and generate predictions:
    ```bash
    jupyter notebook main.ipynb
    ```

The predictions for the unknown images will be logged in `predictions_log.txt`.
