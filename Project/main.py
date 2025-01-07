from helpers import *
from unet import *
from keras.metrics import MeanIoU

import matplotlib.pyplot as plt

train_images, train_masks, test_images, test_masks = prepare_images()
size = 256
batch = 8

# Create the U-Net model
model = unet_model(input_size=(256, 256, 1))

# Print model summary
model.summary()

# Model Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',MeanIoU(num_classes=2,name="IOU")]
)

# Training
results = model.fit(
    train_images, train_masks,
    validation_data = (test_images,test_masks),
    epochs=5,
    batch_size=batch
)

fig, ax = plt.subplots(1, 2, figsize=(10,3))
ax[0].plot(results.epoch, results.history["loss"], label="Train loss")
ax[0].plot(results.epoch, results.history["val_loss"], label="Validation loss")
ax[0].legend()
ax[1].plot(results.epoch, results.history["accuracy"], label="Train accuracy")
ax[1].plot(results.epoch, results.history["val_accuracy"], label="Validation accuracy")
ax[1].legend()
fig.suptitle('Loss and Accuracy', fontsize=16)
plt.show()


fig, ax = plt.subplots(5,3, figsize=(10,18))
test_images,test_masks

j = np.random.randint(0,test_images.shape[0], 5)
for i in range(5):
    ax[i,0].imshow(test_images[j[i]])
    ax[i,0].set_title('Image')
    ax[i,1].imshow(test_masks[j[i]],cmap='gray')
    ax[i,1].set_title('Mask')
    y_hat = model.predict(np.expand_dims(test_images[j[i]],0),verbose=0)[0]
    threshold = 0.5
    binary_mask = (y_hat > threshold).astype(np.uint8)
    ax[i,2].imshow(binary_mask,cmap='gray')
    ax[i,2].set_title('Prediction')
fig.suptitle('Results', fontsize=16)
plt.show()  