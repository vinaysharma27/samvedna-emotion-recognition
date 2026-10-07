
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
print(f"Confusion matrix saved to {conf_matrix_file}")
