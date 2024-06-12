import tensorflow as tf
#from tensorflow.keras import layers, models
import numpy as np
from keras import layers, models, datasets
#import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import pandas as pd
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

def exploracion(imagenes, etiquetas, nombre):
    datos = pd.DataFrame({
    'label': etiquetas.flatten(),
    'image': [imagenes[i] for i in range(imagenes.shape[0])]
    }) 

    print(f"\n///////////////////// Descripcion {nombre} /////////////////\n{datos.describe()}\n\n")
    print(f"/////// Conteo\n{datos.value_counts('label')}\n\n")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=datos, palette='viridis')
    plt.title(f'Distribución de etiquetas en el dataset de {nombre}')
    plt.savefig(f'distribucion_etiquetas_{nombre}.png')
    #plt.clf()
    plt.show()
    
    #return datos

def cargarMAT(archivo):  # ya sirve
    data = sio.loadmat(archivo)
    #formateo de datos del .mat
    imagenes = np.transpose(data['X'], (3, 0, 1, 2))
    etiquetas = data['y'].flatten()
    # Reemplazar el label 10 por 0 para dígitos
    etiquetas[etiquetas == 10] = 0
    return imagenes, etiquetas


def preparar(train_images, test_images, train_labels, test_labels):
    #normalizacion
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    x_train = np.expand_dims(train_images, axis=-1)
    x_test = np.expand_dims(test_images, axis=-1)

    train_labels = train_labels.reshape(-1,1)
    test_labels = test_labels.reshape(-1,1)
    
    return x_train, x_test, train_labels, test_labels


def entrenar(elec):
    #carga cifar10
    if elec == 1:
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    else:
        #carga svhn
        train_images, train_labels = cargarMAT('train_32x32.mat')
        test_images, test_labels = cargarMAT('test_32x32.mat')

    exploracion(train_images, train_labels, "entrenamiento")

    x_train, x_test, train_labels, test_labels = preparar(train_images, test_images, train_labels, test_labels)
    
    imprimirImagenes(x_train, train_labels, elec, "imagenes_entrenamiento")
    #print(train_labels)

    # El modelo sera una red convolucional para poder trabajar mejor con clasificacion de imagenes

     #input: altura x ancho x numero de mapas    Parametros: (dimensiones filtro x mapas de entrada + 1) x num filtros

    model = models.Sequential([  #input 32x32 + 3 RGB  
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='valid'), #salida 30 x 30 x 32, Parametros=(3x3x3+1)*32 = 896
    layers.MaxPooling2D((2, 2)),           #salida 15 x 15 x 32
    layers.Conv2D(64, (3, 3), activation='relu', padding='valid'), # salida 13 x 13 x 64, Parametros (3 x 3 x 32 + 1) * 64 = 18,496 
    layers.MaxPooling2D((2, 2)),   # salida 6 x 6 x  64
    layers.Conv2D(64, (3, 3), activation='relu', padding='valid'), # salida 4 x 4 x 64, Parametros (3 x 3 x 64 + 1)* 64 = 36,928
    layers.Flatten(),  # salida  8 x 8 x 64 = 4096
    layers.Dense(64, activation='relu'),  # Parametros (4096 + 1 ) * 64 = 262, 208
    layers.Dense(10, activation='softmax')  # 10 neuronas. Una por clase Parametros (64 + 1)* 10
    ])


    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    print("\n////// A entrenar ////////\n")
    
    entrenado = model.fit(x_train, train_labels, epochs=10, validation_data=(x_test, test_labels))
    
    model.save('modelo.h5')

    plt.plot(entrenado.history['accuracy'], label='accuracy')
    plt.plot(entrenado.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Generacion')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('historial_accuracy.png')
    #plt.clf()
    plt.show()

    plt.plot(entrenado.history['loss'], label='loss')
    plt.plot(entrenado.history['val_loss'], label = 'val_loss')
    plt.xlabel('Generacion')
    plt.ylabel('Perdida')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('historial_loss.png')
    #plt.clf()
    plt.show()

    print("\n////// A probar ////////\n")
    probar(model, x_test, test_labels, elec)


def probar(model, x_test, test_labels, elec):
    predicciones= model.predict(x_test)
    # Aplicar softmax a las predicciones
    predicted_probabilities = tf.nn.softmax(predicciones, axis=1)
    # Obtener el índice de la clase con la mayor probabilidad
    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    print(predicted_labels)
    predicted_labels = predicted_labels.reshape(-1,1)
    #np_etiquetasP = np.array(predicted_labels)
    #print(predicted_labels)

    imprimirImagenes(x_test, predicted_labels, elec, "imagenes_etiquetas_modelo")
    exploracion(x_test, predicted_labels, "modelo")

    imprimirImagenes(x_test, test_labels, elec, "imagenes_etiquetas_prueba")
    exploracion(x_test, test_labels, "prueba")
    #evalucacion
    test_loss, test_acc = model.evaluate(x_test, test_labels)
    print(f"test_acc: {test_acc}")
    print(f"test_loss: {test_loss}")


def imprimirImagenes(imagenes, etiquetas, elec, nombre, num_filas=8, num_columnas=8):  # ya sirve
    if elec==1:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']   #van del 0 al 9
    else:
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 

    fig, axes = plt.subplots(num_filas, num_columnas, figsize=(20, 20))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        imagen = np.squeeze(imagenes[i])
        ax.imshow(imagen)
        ax.set_title(class_names[etiquetas[i][0]])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{nombre}.png')
    #plt.clf()
    plt.show()


if __name__ == "__main__":

    direccion = "modelo.h5"
    if os.path.exists(direccion):
        print("\n Ya existe un modelo entrenado\n ")
        modelo = models.load_model(direccion)
        
        train_images, train_labels = cargarMAT('test_32x32.mat')
        test_images, test_labels = cargarMAT('test_32x32.mat')

        x_train, x_test, train_labels, test_labels = preparar(train_images, test_images, train_labels, test_labels)

        probar(modelo, x_test, test_labels, 2)

    else:
        user_input = int(input("Eleige dataset (1 - cifar10, 2 - SVHN): "))
        entrenar(user_input)