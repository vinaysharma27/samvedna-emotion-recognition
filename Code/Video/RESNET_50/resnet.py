import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, ZeroPadding2D, Flatten, Dense, AveragePooling2D
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import classification_report
import pandas as pd

# Set image data format for TensorFlow
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

# Define dataset path
dataset_path = '----'  # New dataset path
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']  # 6 emotion labels

def load_images_from_video_folder(dataset_path, batch_size=64, frame_skip=10):
    records = []

    for emotion in emotion_labels:
        emotion_folder = os.path.join(dataset_path, emotion)

        if not os.path.isdir(emotion_folder):
            print(f"Warning: Emotion folder '{emotion}' not found, skipping...")
            continue

        for video_file in os.listdir(emotion_folder):
            video_path = os.path.join(emotion_folder, video_file)

            # Skip non-video files (if any)
            if not video_file.endswith('.mp4'):
                continue

            # Same video ID structure as the previous video-level code
            video_id = f"{emotion}/{os.path.splitext(video_file)[0]}"

            try:
                # Try to open the video file
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Warning: Could not open video file {video_file}, skipping...")
                    continue

                frame_count = 0  # Frame counter
                # Read frames from the video and extract images
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # No more frames to read

                    frame_count += 1
                    # Process only every 10th frame
                    if frame_count % frame_skip == 0:
                        # Resize the frame to the desired size (e.g., 48x48)
                        frame_resized = cv2.resize(frame, (48, 48))
                        frame_resized = frame_resized / 255.0  # Normalize to [0, 1]

                        # Store the frame, label, and video ID
                        records.append({
                            "image": frame_resized,
                            "label": emotion_labels.index(emotion),
                            "video_id": video_id,
                        })

                cap.release()  # Release the video capture object
            except Exception as e:
                print(f"Error processing video {video_file}: {e}, skipping...")
                continue

    return records


# Load all frame records
records = load_images_from_video_folder(dataset_path)




video_to_label = {}

for record in records:
    video_id = record["video_id"]
    label = int(record["label"])

    if video_id in video_to_label:
        if video_to_label[video_id] != label:
            raise RuntimeError(
                f"Video ID '{video_id}' has inconsistent labels."
            )
    else:
        video_to_label[video_id] = label

unique_video_ids = np.asarray(list(video_to_label.keys()))

unique_video_labels = np.asarray(
    [video_to_label[v] for v in unique_video_ids],
    dtype=np.int64,
)

train_video_ids, test_video_ids = train_test_split(
    unique_video_ids,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=unique_video_labels,
)

train_set = set(train_video_ids)
test_set = set(test_video_ids)

if train_set & test_set:
    raise RuntimeError("Video-level data leakage detected.")

train_records = [
    r for r in records if r["video_id"] in train_set
]

test_records = [
    r for r in records if r["video_id"] in test_set
]

train_images = np.asarray(
    [r["image"] for r in train_records],
    dtype=np.float32,
)
train_labels = np.asarray(
    [r["label"] for r in train_records],
    dtype=np.int64,
)

test_images = np.asarray(
    [r["image"] for r in test_records],
    dtype=np.float32,
)
test_labels = np.asarray(
    [r["label"] for r in test_records],
    dtype=np.int64,
)

# Keep the original variable names used by the training code
X_train = train_images
X_test = test_images
Y_train = train_labels
Y_test = test_labels

# Print dataset info
print(f"Number of unique videos: {len(unique_video_ids)}")
print(f"Number of training videos: {len(train_video_ids)}")
print(f"Number of test videos: {len(test_video_ids)}")
print(f"Number of training examples: {X_train.shape[0]}")
print(f"Number of test examples: {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=len(emotion_labels))
Y_test = to_categorical(Y_test, num_classes=len(emotion_labels))

# Plot a few images
def plot_images(images_arr, labels):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))  # Plot 10 images at a time
    axes = axes.flatten()
    for ax, img, label in zip(axes, images_arr, labels):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(emotion_labels[np.argmax(label)])  # Display emotion label
    plt.tight_layout()
    plt.show()

plot_images(X_train[:10], Y_train[:10])  # Plot first 10 images from the training set

from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation
from tensorflow.keras.initializers import glorot_uniform

def identity_block(X, f, filters, stage, block):
    '''
    Implementation of identity block described above

    Arguments:
    X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -       defines the shape of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention

    Returns: 
    X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
    '''

    # defining base name for block
    conv_base_name = 'res' + str(stage) + block + '_'
    bn_base_name = 'bn' + str(stage) + block + '_'

    # retrieve number of filters in each layer of main path
    f1, f2, f3 = filters

    # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
    bn_axis = 3  # this is because images typically have the shape (batch_size, height, width, channels)

    # save input for "addition" to last layer output; step in skip-connection
    X_skip_connection = X

    # ----------------------------------------------------------------------
    # Building layers/components of identity block using Keras functional API

    # First component/layer of main path
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_base_name+'first_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
    X = Activation('relu')(X)

    # Second component/layer of main path
    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_base_name+'second_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
    X = Activation('relu')(X)

    # Third component/layer of main path
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_base_name+'third_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)

    # "Addition step" - skip-connection value merges with main path
    # NOTE: both values have the same dimensions at this point, so no operation is required to match dimensions
    X = Add()([X, X_skip_connection])
    X = Activation('relu')(X)

    return X


# In[6]:


from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation
from tensorflow.keras.initializers import glorot_uniform

def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in the ResNet architecture.

    Arguments:
    X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -       defines the shape of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention
    s -       specifies the stride to be used (default is 2)

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Defining base name for block
    conv_base_name = 'res' + str(stage) + block + '_'
    bn_base_name = 'bn' + str(stage) + block + '_'
    
    # Retrieve number of filters in each layer of the main path
    f1, f2, f3 = filters
    
    # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case.
    bn_axis = 3

    # Save input for "addition" to last layer output (skip-connection)
    X_skip_connection = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(f1, (1, 1), strides=(s, s), padding='valid', name=conv_base_name+'first_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(f2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_base_name+'second_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_base_name+'third_component',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)

    ##### Skip Connection #####
    # Convolve skip-connection value to match its dimensions to the third layer output's dimensions
    X_skip_connection = Conv2D(f3, (1, 1), strides=(s, s), padding='valid', name=conv_base_name+'merge',
                                kernel_initializer=glorot_uniform(seed=0))(X_skip_connection)
    X_skip_connection = BatchNormalization(axis=3, name=bn_base_name+'merge')(X_skip_connection)

    # "Addition step" - Add the skip connection to the output of the main path
    X = Add()([X, X_skip_connection])
    X = Activation('relu')(X)

    return X


# In[7]:


from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    ResNet50 architecture as described in the paper
    Arguments:
    input_shape - shape of the images of the dataset
    classes - number of classes (emotions in your case)
    
    Returns:
    model - a Keras Model() instance
    """
    
    # Input layer
    X_input = Input(input_shape)

    # Zero-Padding: pad the input image with 3x3 padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1: Initial Convolution + MaxPooling
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv_1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # Flatten and output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Create the model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

# Number of classes (emotions)
classes = 6  # 6 emotion labels: Anger, Disgust, Fear, Happy, Neutral, Sad

# Define and compile the model
input_shape = (48, 48, 3)
model = ResNet50(input_shape=input_shape, classes=classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the training history to an Excel file
history_df = pd.DataFrame(history.history)
history_df.to_excel('D:/training_history.xlsx', index=False)

# Save the trained model
model.save('emotion_recognition_model.h5')

# Generate and save classification report
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

report = classification_report(Y_test_classes, Y_pred_classes, target_names=emotion_labels)
with open('D:/classification_report.txt', 'w') as f:
    f.write(report)

# Plot the training/validation accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig('D:/training_plot.png')
plt.show()

desktop_path = os.path.expanduser('~') + '/Desktop/'
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
Y_pred_classes = np.argmax(Y_pred, axis=1)  # Convert predictions to class indices
Y_true_classes = np.argmax(Y_test, axis=1)  # True class indices

conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)

# Plot the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the confusion matrix plot to the desktop
conf_matrix_file = os.path.join(desktop_path, 'confusion_matrix.png')
plt.tight_layout()
plt.savefig(conf_matrix_file)
plt.close()  # Close the plot to avoid display issues
print(f"Confusion matrix saved to {conf_m
