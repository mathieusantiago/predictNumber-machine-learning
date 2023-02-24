import tensorflow as tf
import numpy as np

def predictNumbre():
    # Définir les données d'entrée et de sortie
    x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_train = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    # Créer un modèle séquentiel
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]))
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Compiler le modèle
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(x_train, y_train, epochs=100000000)

    # Utiliser le modèle pour prédire un nombre
    number_to_guess = 5
    prediction = model.predict([number_to_guess])
    if prediction > 0.5:
        print(f"Le modèle prédit que le nombre {number_to_guess} est 5.")
    else:
        print(f"Le modèle prédit que le nombre {number_to_guess} n'est pas 5.")
    print(f"la prediction été de {prediction[0][0]}")

predictNumbre()