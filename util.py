def load_data_labels(raw_labels, base_loc, verbose=False):
    """
    Function that accepts a list of 'filename, angle' strings,
    a path object pointing to the folder containing images and 
    a verbose flag to control print verbosity
    
    Returns a list of images (numpy array in BGR channel-ordering) 
    and angles (int)
    """
    import cv2
    from tqdm import tqdm
    
    # Initialize empty lists to store image & angles
    imgs_l = []
    angles_l = []
    
    # Iterate over each row of the labels file
    for raw_label in tqdm(raw_labels):

        # Split each line by comma followed by a space
        combined = raw_label.split(', ')
        
        # Generate file path from the filename
        # Extract both the file path as well as the angle
        filepath = str(base_loc/combined[0])
        angle = int(combined[1])
        
        # Use OpenCV to load the file
        img = cv2.imread(filepath)
        
        # Append the image & angle to respective lists
        imgs_l.append(img)
        angles_l.append(angle)
    
    if verbose:
        
        print(f'[INFO] Found {len(imgs_l)} elements in the set!')    
        print('[INFO] Printing a few (X, y) pairs:')
        
        # # Printing 5 images randomly from the list
        # for i in range(5):  

        #     rand_idx = np.random.choice(range(len(imgs_l)))
        #     rand_X = imgs_l[rand_idx]
        #     rand_y = angles_l[rand_idx]

        #     # Add the angle value as text on the image for display
        #     cv2.putText(rand_X, f"Angle: {rand_y}", (10, 30), 0, 0.7, (0, 0, 255), 2)
        
        #     if COLAB:

        #         # Colab does not allow opening of windows
        #         # Use the official patch to display image in IPython in Colab
        #         cv2_imshow(rand_X)
            
        #     else:
        #         cv2.imshow(f'Image #{rand_idx + 1}', rand_X)
        #         cv2.waitKey(0)
        
    return imgs_l, angles_l

def preprocess_image(image, resize, flatten=True):

  """
  Function that accepts an image (numpy array) and a flatten (bool)
  flag to return a 1D vector instead of a 2D matrix

  Returns a preprocessed image
  """
  import cv2

  # Use Gaussian Blur with a 3x3 filter to smoothen out/denoise
  blur = cv2.GaussianBlur(image, (3, 3), 0)

  # Resize the image to a square
  resized_img = cv2.resize(blur, (resize, resize))

  # Convert the image to grayscale - flip bits to change
  # foreground to background and vice versa to boost textual areas
  gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
  gray = cv2.bitwise_not(gray)

  # Use adaptive thresholding (Otsu's technique) to create
  # a high-contrast threshold map
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  
  # Use the Laplacian filter
  laplacian = cv2.Laplacian(thresh, cv2.CV_8UC3)
  
  # Convert 2D image to 1D vector for modelling
  if flatten:
    laplacian = laplacian.reshape((-1))
  
  return laplacian

def preprocess_dataset(images_l, angles_l, resize, flatten=True, shuffle=False):
  
  """
  Function that accepts a list of images (numpy arrays) and angles (int)
  as well as 2 boolean flags — flatten to determine whether a 1D vector
  or a 2D matrix is returned and shuffle to randomise the order of train data

  Returns (data, labels)
  """
  import numpy as np
  from tqdm import tqdm

  # Initialize empty list to store preprocessed images
  data = []

  # Iterate through each image
  for image in tqdm(images_l):

    # Preprocess the image
    processed_img = preprocess_image(image, resize, flatten)
    
    # Append the preprocessed image to the list
    data.append(processed_img)

  # Convert angles to class labels (0, 1, 2, 3)
  y = [int(angle/90) for angle in angles_l]
  
  # Convert lists to arrays
  data = np.array(data)
  y = np.array(y)

  # Shuffle both the images and the labels if required
  if shuffle:
    rng = np.random.default_rng(42)
    el_idx = [*range(len(data))]
    rng.shuffle(el_idx)
    data = data[el_idx]
    y = y[el_idx]

  # Return a tuple of 2 numpy arrays
  return (data, y)

def fit_automl(train_X, train_y, pca=False):

  """
  Function that uses AutoML to fit a classification model
  on training data (with dimensionality reduced by PCA if required) 
  and labels

  Returns the fit model and pca object if required
  """
  from flaml import AutoML
  from sklearn.metrics import accuracy_score
  from sklearn.decomposition import PCA

  # Use FLAML's AutoML for both Lvl 1 & 2 model selection
  automl = AutoML()

  if pca:
    
    # PCA with 85% variance
    pca_object = PCA(n_components=0.85)
    train_X = pca_object.fit_transform(train_X)

  automl.fit(train_X, train_y, task="classification", time_budget=60)
  
  # Print the performance on the training data
  print(f"[INFO] Accuracy on train data: {accuracy_score(automl.predict(train_X), train_y)}")

  if pca:
    return (automl, pca_object)
  else:
    return automl

def enrich_data(imgs_l, angles_l):
  
  """
  Function that accepts a list of images (numpy arrays) and angles (int)

  Returns a tuple of enriched data (numpy array) and corresponding labels (int)
  """    
  import itertools
  import cv2
  from tqdm import tqdm

  # Create new lists to store the images & labels
  rot_imgs_l  = []
  rot_labels_l = []
  ANGLES_L = [0, 90, 180, 270]

  # Loop over each image in training data
  for img, label in tqdm(list(zip(imgs_l, angles_l))):

    # Append original image & label to respective lists
    rot_imgs_l.append(img)
    rot_labels_l.append(label)

    # To extract new labels for each successive rotation
    # Create an iterator that can cycle through the 4 angle values
    cycling = itertools.cycle(ANGLES_L)
    
    # Move iterator to the index of current image angle
    skipping = itertools.dropwhile(lambda x : x != label, cycling)
    
    # Create an iterator slice starting from position after current angle 
    slicing = itertools.islice(skipping, 1, len(ANGLES_L))

    # Use a loop to rotate image 90 three times while altering the label
    for new_angle in slicing:
      
      # Rotate the image by 90 degrees
      img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

      # Append the rotated image to rot_imgs_l
      rot_imgs_l.append(img)

      # Append the label to rot_labels_l
      rot_labels_l.append(new_angle)

  return (rot_imgs_l, rot_labels_l)