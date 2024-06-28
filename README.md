### A. Image Preprocessing Steps

1. Sharpening of Image pixels (Increase the image quality):
   
   The purpose is to enhance edges and details in images, potentially improving feature extraction.
   Used techniques like filtering on fixed size kernel that applies transformation and brightens the image to emphasize edges and details. <br/>

3. Color Conversion (Ignore the color pixles and focus on edges and patterns):

    Basically, we need to convert images to a suitable color space based on task requirements.<br/>
    Convert images to grayscale or other color spaces, this step will reduce computational complexity and focuses on the relevant image features.

### B. Importance of Selected Feature Sets for Image Classification

The selected feature sets are (HOG for local patterns, Canny for edge detection, and SIFT for textures) 

1. Local Patterns Using Histograms

   - Histograms of oriented gradients (HOG) capture local patterns of gradients in images.
   - It is particularly useful for distinguishing between different textures and patterns present in images.
   - This will help in capturing different distribution of gradients in forests as compared to other classes.

2. Edge Detection Using Canny

   - This algorithm helps in identifying the sharp changes in intensity and boundaries within images.
   - For example this can help in differentiating between the buildings, and other classes like mountains.

3. Textures Using SIFT (Scale-Invariant Feature Transform)

   - SIFT features capture distinctive local textures and patterns invariant to scale and rotation.
   - This is beneficial for tasks where texture differences, such as those between glaciers and seas or forests and mountains, are crucial for classification.

### C. Apply appropriate techniques for dimensionality reduction, if your feature set size is too large and explain that

While I didn't focused on dimensional reductionality on features but have done resampling to ensure there are imbalance classes between datasets.

### D. Evaluation of the trained models using appropriate metrics.

1. Metrics:

Precision: The ratio of correctly predicted positive observations to the total predicted positives. <br/>

```
Precision = TP/(TP+FP)
```

Recall: The ratio of correctly predicted positive observations to all the observations in the actual class. <br/>

```
Recall = TP/(TP+FN)
```
F1-score: The weighted average of Precision and Recall. <br/>

```
F1 Score = 2*(Recall * Precision) / (Recall + Precision)
```

Support: The number of actual occurrences of the class in the dataset.<br/>

![image](https://github.com/abhishekY2401/image_classification/assets/89199323/732e5b41-b2cf-4641-8c13-395c7001aaa7)

![image](https://github.com/abhishekY2401/image_classification/assets/89199323/a6f6bac6-eab2-4d94-b028-aecb30c58292)

### E. Algorithms Used

1. K Nearest Neighbors (accuracy was near to ~ 54%, after hyper parameter tuning it increase to ~ 84%)
2. Random Forest (accuracy was near to ~ 53%)

### F. Development of Flask application

So, essentially we have one ```/classification``` route which pre-processes the image and extracts relevant features and then feeds it to our KNN classification model.

1. pre-processing of image -> sharpening of image (increasing the image quality) -> convert to gray pixels
2. feature extraction -> {local patterns, edges, texture}
3. pass the features to model for prediction.

### G. Setting up flask application

1. Clone the repository 
2. Install virtual environment and activate it 

   ```
    python -m venv venv
   ./venv/Scripts/activate
   ```
   
3. Install the dependencies

   ```
   pip install -r requirements.txt
   ```

4. Finally run the flask application

    ```
    python app.py
    ```

### H. Enhancement of Model and Feature Extraction Automation

1. Advance feature engineering using deep learning techniques like pre trained CNN models which can capture more details about the images
2. Data augmentation and expansion, increasing different classes in datasets would help in training the model on varied amount of data.

Feature Extraction Process

1. Define scripts using custom functions of OpenCV for feature extraction for images.
2. Containerize feature extraction process using docker and deploy on cloud services like AWS EC2.


