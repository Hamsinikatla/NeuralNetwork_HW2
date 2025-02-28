# NeuralNetwork_HW2





# 2 Convolution Operations with Different Parameters

## Overview
This program demonstrates how convolution operations are performed on a 5x5 input matrix using a 3x3 kernel with different strides and padding types. The `scipy.signal.convolve2d` function is used to compute the 2D convolution. The code explores two types of padding (`'valid'` and `'same'`) and two different stride values (1 and 2), producing output feature maps for each combination.

## Installation Instructions
Before running the code, ensure that you have Python installed along with the required libraries: `numpy` and `scipy`. You can install the necessary libraries using pip:


pip install numpy scipy


## Usage Guide
To use the script:
1. Ensure that the Python environment has the required dependencies installed.
2. Copy the provided code into a Python script file (e.g., `convolution_operations.py`).
3. Run the script using Python:


python convolution_operations.py


The program will output the results of the convolution operations for different stride and padding settings.

## Code Explanation Steps
1. **Input Matrix**: A 5x5 matrix is defined as the input. This is the matrix that will undergo convolution with the kernel.
2. **Kernel**: A 3x3 kernel is defined, which will be used for the convolution operation.
3. **apply_convolution function**: This function accepts two parameters—`stride` and `padding`—to apply the convolution with the given stride and padding. If padding is `'same'`, the input matrix is padded with zeros; otherwise, the input is used as is.
4. **Convolution Process**: The `convolve2d` function from `scipy.signal` is used to apply the kernel to the input matrix, followed by downsampling based on the stride value.
5. **Results**: The results for different combinations of stride and padding are stored in a dictionary and printed.

## Output
The script prints the convolution results for the following combinations of stride and padding:

1. **Stride 1, Padding VALID**: This means no padding is applied and the output feature map is smaller than the input matrix.
2. **Stride 1, Padding SAME**: This uses zero-padding to maintain the size of the output feature map the same as the input.
3. **Stride 2, Padding VALID**: This uses stride 2 (downsampling the output) with no padding.
4. **Stride 2, Padding SAME**: This uses stride 2 with zero-padding to maintain the size of the output feature map as much as possible.

Example output:

```
Stride 1, Padding VALID:
[[ -6.  -6.  -6.]
 [ -6.  -6.  -6.]
 [ -6.  -6.  -6.]]

Stride 1, Padding SAME:
[[  1.   3.   3.   3.   5.]
 [  6.   4.  -4.  -4.  10.]
 [ 11.  12.  13.  14.  15.]
 [ 16.  18.  19.  19.  20.]
 [ 21.  23.  24.  24.  25.]]

Stride 2, Padding VALID:
[[ -6.  -6.]
 [ -6.  -6.]]
 
Stride 2, Padding SAME:
[[ 1.  3.]
 [ 6.  4.]]
```

## Summary of Outputs
- **Stride 1, Padding VALID**: The output is smaller than the input matrix due to the lack of padding. The convolution results are primarily negative due to the kernel configuration.
- **Stride 1, Padding SAME**: The output retains the same size as the input matrix, with zero-padding applied. The values in the output reflect the convolution results with padding.
- **Stride 2, Padding VALID**: The output is downsampled by a factor of 2, resulting in a smaller matrix.
- **Stride 2, Padding SAME**: The output is also downsampled by a factor of 2, with zero-padding applied to maintain the output matrix size as much as possible.

## Key Learnings
- The **stride** parameter controls how much the output is downsampled (higher stride = more downsampling).
- The **padding** parameter affects the size of the output matrix. `'same'` padding ensures the output is the same size as the input, while `'valid'` padding reduces the output size.
- The values in the output depend on the kernel's values and the input matrix, as well as the combination of stride and padding.

## Features
- Supports different **stride** values (1 and 2).
- Allows two types of **padding**: `'valid'` and `'same'`.
- Easy to modify for different input matrices or kernel sizes.
- Uses **SciPy's `convolve2d`** for efficient 2D convolution operations.










# 3 CNN Feature Extraction with Filters and Pooling

## Overview
This program demonstrates two key operations used in Convolutional Neural Networks (CNNs): **edge detection** using the Sobel filter and **pooling** (both **max pooling** and **average pooling**) using NumPy. The Sobel filter detects edges in an image, while pooling operations help in reducing the dimensionality of data, retaining important features. This code also showcases the use of OpenCV for image processing and NumPy for pooling operations.

## Installation Instructions
Ensure that you have the required libraries: **NumPy** and **OpenCV** (for Sobel filtering). You can install the necessary libraries using pip:


pip install numpy opencv-python


Make sure you also have a valid image file to test the Sobel filter, e.g., `sample_image.jpg`.

## Usage Guide
### 1. Sobel Filter for Edge Detection
To apply the Sobel filter to an image:
1. Download or prepare a grayscale image.
2. Replace `"sample_image.jpg"` in the script with the path to your image file.
3. Run the script using Python:


python cnn_feature_extraction.py


The script will display the original image and the edge-detected results using the Sobel filter (both in the x and y directions).

### 2. Max Pooling and Average Pooling
For max pooling and average pooling operations:
- The script generates a random 4x4 matrix to demonstrate the pooling process.
- The size of the pooling window is defined as `2x2` (can be adjusted by changing the `pool_size`).
- You can modify the input matrix as needed.

The pooling results will be printed in the terminal.

## Code Explanation Steps
### Task 1: Edge Detection Using Sobel Filter
1. **Load Image**: The image is loaded as a grayscale image using OpenCV (`cv2.imread`).
2. **Sobel Filter**: The Sobel filter is applied in both the **x** and **y** directions using `cv2.Sobel`. This helps to detect edges in the horizontal and vertical directions.
3. **Display Results**: The original image and Sobel-filtered images are displayed using OpenCV’s `cv2.imshow`.

### Task 2: Max Pooling and Average Pooling using NumPy
1. **Max Pooling**:
   - Divides the input matrix into non-overlapping pools (windows) of size `pool_size x pool_size`.
   - The maximum value from each pool is selected to form the output matrix.
2. **Average Pooling**:
   - Similar to max pooling, but instead of selecting the maximum value, the mean of the values in each pool is calculated.
3. **Output**: The results of the pooling operations (max pooled and average pooled) are printed in the terminal.

### Example Matrix:
A random 4x4 matrix is generated, and both max pooling and average pooling operations are performed on it.


input_matrix = np.random.rand(4, 4).astype(np.float32)


## Output
The output consists of:
1. **Sobel Filter Output**:
   - Three windows will appear showing the original image, Sobel-X, and Sobel-Y edge-detection results.
   - The Sobel-X filter detects edges in the horizontal direction, while Sobel-Y detects edges in the vertical direction.

2. **Pooling Results**:
   - The original 4x4 matrix is displayed, followed by the **max pooled** and **average pooled** matrices.
   Example output for a random 4x4 matrix:
   
   ```
   Original Matrix:
   [[0.69728284 0.37365062 0.20852872 0.68604385]
    [0.24868114 0.06341324 0.55304244 0.69486899]
    [0.80401669 0.21852257 0.46584783 0.76360369]
    [0.32764777 0.88948269 0.4361238  0.15828617]]

   Max Pooled Matrix:
   [[0.69728284 0.68604385]
    [0.80401669 0.76360369]]

   Average Pooled Matrix:
   [[0.58654763 0.55230453]
    [0.62420643 0.59825233]]
   ```

## Summary of Outputs
- **Sobel Filter Output**: 
  - Displays the edge-detection results for the x and y directions.
  - Provides insights into the horizontal and vertical gradients of the input image.
  
- **Pooling Outputs**:
  - **Max Pooling**: Retains the highest value from each pool.
  - **Average Pooling**: Retains the average value from each pool.

## Key Learnings
- **Edge Detection**: The Sobel filter is a simple yet effective tool for detecting edges in images. The direction of edges (horizontal or vertical) can be identified by applying the Sobel filter in respective directions.
- **Pooling Operations**: Pooling helps in reducing the spatial dimensions of the input matrix, making it computationally efficient while preserving important features. 
  - **Max Pooling** focuses on the most prominent features (maximum values).
  - **Average Pooling** provides a smoother feature map by averaging over the pooled values.

## Features
- **Edge Detection**: Uses the Sobel filter to detect edges in images in both horizontal and vertical directions.
- **Pooling**: Implements both max pooling and average pooling for dimensionality reduction.
- **Flexible Input**: You can modify the input image for Sobel filtering or adjust the pooling window size and input matrix.
- **OpenCV and NumPy Integration**: Leverages OpenCV for image manipulation and NumPy for efficient pooling operations.







# 4 Implementing and Comparing CNN Architectures

## Overview
This program demonstrates the implementation of two well-known Convolutional Neural Network (CNN) architectures: **AlexNet** and **ResNet**. It shows how to define and create both models using TensorFlow and Keras, along with key components such as convolutional layers, pooling layers, residual blocks, and fully connected layers. These models are widely used in image classification tasks.

- **AlexNet**: A deep CNN with several convolutional layers followed by fully connected layers and dropout for regularization.
- **ResNet**: A CNN with residual blocks that allow the network to skip connections, helping with the vanishing gradient problem and improving training efficiency.

## Installation Instructions
Before running the script, make sure you have **TensorFlow** installed in your Python environment. You can install it using pip:


pip install tensorflow


Ensure you are using Python 3.6 or higher for compatibility with TensorFlow.

## Usage Guide
To use this script:
1. Install the required libraries as mentioned above.
2. Copy the provided code into a Python script file (e.g., `cnn_comparison.py`).
3. Run the script to see the architectures defined and their summaries printed in the console:


python cnn_comparison.py


The model summaries for both **AlexNet** and **ResNet** will be printed, showing the layers and parameters for each architecture.

## Code Explanation Steps

### Task 1: AlexNet
- **AlexNet Architecture**:
  - **Conv2D**: Convolutional layers with ReLU activation.
  - **MaxPooling2D**: Max pooling layers for downsampling.
  - **Dropout**: Dropout layers to prevent overfitting.
  - **Dense**: Fully connected layers, with the last one using a `softmax` activation for multi-class classification.
  - The model is designed to accept an input image of shape (227, 227, 3) and outputs predictions for 10 classes.

```
def create_alexnet():
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(256, (5, 5), activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model
```

### Task 2: ResNet
- **ResNet Architecture**:
  - **Residual Block**: Contains two convolutional layers with a skip connection that adds the input tensor to the output of the block, improving gradient flow.
  - The ResNet model starts with a large convolutional layer, followed by multiple residual blocks, and ends with a fully connected layer that outputs predictions for 10 classes.
  
```
def residual_block(input_tensor, filters):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = Add()([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x
```

### Summary of Model Creation:
- **AlexNet**: Defined using the Sequential API, stacking layers in order.
- **ResNet**: Defined using the Functional API, with a custom residual block function that allows for skip connections.

## Output
When the script is executed, you will see the following outputs printed to the console:

1. **AlexNet Model Summary**:
   ```
   Model: "sequential"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   conv2d (Conv2D)              (None, 54, 54, 96)        34944     
   max_pooling2d (MaxPooling2D) (None, 26, 26, 96)        0         
   conv2d_1 (Conv2D)            (None, 22, 22, 256)       614656    
   max_pooling2d_1 (MaxPooling2 (None, 10, 10, 256)       0         
   conv2d_2 (Conv2D)            (None, 8, 8, 384)         885120    
   conv2d_3 (Conv2D)            (None, 6, 6, 384)         1327488   
   conv2d_4 (Conv2D)            (None, 4, 4, 256)         884992    
   max_pooling2d_2 (MaxPooling2 (None, 2, 2, 256)         0         
   flatten (Flatten)            (None, 1024)              0         
   dense (Dense)                (None, 4096)              4198400   
   dropout (Dropout)            (None, 4096)              0         
   dense_1 (Dense)              (None, 4096)              16781312  
   dropout_1 (Dropout)          (None, 4096)              0         
   dense_2 (Dense)              (None, 10)                40970     
   =================================================================
   Total params: 24,091,882
   Trainable params: 24,091,882
   Non-trainable params: 0
   ```

2. **ResNet Model Summary**:
   ```
   Model: "model"
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #   
   =================================================================
   input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
   conv2d (Conv2D)              (None, 112, 112, 64)      9472      
   conv2d_1 (Conv2D)            (None, 112, 112, 64)      36928     
   add (Add)                    (None, 112, 112, 64)      0         
   activation (Activation)      (None, 112, 112, 64)      0         
   flatten (Flatten)            (None, 802816)            0         
   dense (Dense)                (None, 128)               102758656 
   dense_1 (Dense)              (None, 10)                1290      
   =================================================================
   Total params: 102,804,346
   Trainable params: 102,804,346
   Non-trainable params: 0
   ```

## Summary of Outputs
- **AlexNet** has a total of 24 million parameters, with deep layers for feature extraction followed by large fully connected layers. It is particularly well-suited for large-scale image classification tasks.
- **ResNet** has approximately 102 million parameters and uses residual connections to improve training. This architecture is more efficient in handling deeper networks due to its ability to mitigate the vanishing gradient problem.

## Key Learnings
- **AlexNet**: This architecture uses a series of convolutional layers followed by dense layers for classification. Dropout is used to avoid overfitting.
- **ResNet**: The use of residual blocks makes ResNet more effective for deeper networks, allowing it to skip connections and prevent the vanishing gradient problem.
- **Parameter Count**: AlexNet is significantly smaller compared to ResNet, but ResNet tends to perform better in very deep networks due to residual connections.

## Features
- **AlexNet**: A straightforward deep CNN architecture suitable for large datasets and image classification tasks.
- **ResNet**: A more advanced CNN architecture with residual blocks that improve performance in deeper networks.
- **TensorFlow & Keras**: Both models are implemented using TensorFlow and Keras, leveraging their high-level APIs for easy model definition and training.
