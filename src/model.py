import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def build_leaf_detector():
    # Use pre-trained MobileNetV2 from TensorFlow Hub
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification: leaf or not
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_species_classifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: corn, grape
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_leaf_detector(data):
    model = build_leaf_detector()
    X_train, y_train = data['train']
    X_val, y_val = data['val']

    # Data augmentation for training
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )

    # Fit the generator on training data
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=4),
        epochs=10,
        validation_data=(X_val, y_val),
        steps_per_epoch=max(1, len(X_train) // 4)
    )
    model.save('models/leaf_detector.keras')
    return model, history

def train_species_classifier(data):
    model = build_species_classifier()
    X_train, y_train = data['train']
    X_val, y_val = data['val']

    # Data augmentation for training
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        shear_range=0.2
    )

    # Fit the generator on training data
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=4),
        epochs=20,
        validation_data=(X_val, y_val),
        steps_per_epoch=max(1, len(X_train) // 4)
    )
    model.save('models/species_classifier.keras')
    return model, history

def predict_leaf(image, model):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    return prediction > 0.5  # True if leaf

def predict_species(image, model):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)[0]
    max_prob = np.max(predictions)
    if max_prob < 0.5:  # Low confidence
        return "Unknown leaf"
    class_idx = np.argmax(predictions)
    classes = ['Corn', 'Grape']
    return classes[class_idx]
