import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Cargar CIFAR-10
# ============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

num_classes = 10

# ============================
# 2. Bloque CNN
# ============================
def cnn_block():
    return tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),

        layers.Flatten()
    ])

# ============================
# 3. Bloque MLP profundo
# ============================
def mlp_block():
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu')
    ])

# ============================
# 4. Mini-Transformer
# ============================
def transformer_block(embed_dim=128, num_heads=4, ff_dim=128):
    inputs = layers.Input(shape=(1, embed_dim))

    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(embed_dim)(ff)

    outputs = layers.LayerNormalization()(x + ff)

    return models.Model(inputs, outputs)

# ============================
# 5. Crear modelo RNA-AP completo
# ============================
def build_rna_ap():
    inputs = layers.Input(shape=(32, 32, 3))

    # CNN
    x = cnn_block()(inputs)

    # MLP
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128))(x)

    # Transformer
    transformer = transformer_block()
    x = transformer(x)
    x = layers.Flatten()(x)

    # Clasificador
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# ============================
# 6. Compilar modelo
# ============================
model = build_rna_ap()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================
# 7. Entrenar
# ============================
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64
)

# ============================
# 8. Evaluar
# ============================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Accuracy en test:", test_acc)

# ============================
# 9. Curvas para el paper
# ============================
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy RNA-AP')
plt.xlabel('Épocas')
plt.legend(['Entrenamiento', 'Validación'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida RNA-AP')
plt.xlabel('Épocas')
plt.legend(['Entrenamiento', 'Validación'])
plt.show()
