import cv2
import numpy as np
from skimage.feature import canny
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


# Set a fixed length for the feature vectors
FIXED_FEATURE_LENGTH = 500

# histogram
def compute_histogram(image):
  hist = cv2.calcHist([image], [0], None, [256], [0, 256])
  return hist.flatten()

def edge_detection_canny(image):
    edges = canny(image)
    return edges.flatten()

def compute_sift(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors.flatten() if descriptors is not None else np.array([])

def pad_or_truncate(features, length):
    if len(features) >= length:
        return features[:length]
    else:
        return np.pad(features, (0, length - len(features)), 'constant')
    
def pca_transform(extracted_features):
   pca = PCA(n_components=2, svd_solver='full')
   reduced_features = pca.fit_transform(extracted_features)

   return reduced_features

def sampling(features):
   # Balance the training dataset using SMOTE
    smote = SMOTE(random_state=42)
    feature_resampled = smote.fit_resample(features)

    return feature_resampled
  
# function to extract features from all the images
def extract_features(images):
  features = []

  # loop through each image
  for image in images:
    # new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_features = compute_histogram(image)
    canny_features = edge_detection_canny(image)
    sift_features = compute_sift(image)

    # Debug: print feature shapes
    print("HIST features shape:", hist_features.shape)
    print("Canny features shape:", canny_features.shape)
    print("SIFT features shape:", sift_features.shape)

    # combine the features in different set
    combined_features = np.concatenate([hist_features, sift_features, canny_features])

    # Pad or truncate to the fixed length
    combined_features = pad_or_truncate(combined_features, FIXED_FEATURE_LENGTH)
    print(combined_features.shape)
    features.append(combined_features)

  return np.array(features)

