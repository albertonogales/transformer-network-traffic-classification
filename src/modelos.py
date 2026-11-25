#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "modelos.py".

AUTOR: Carlos Arranz Luque.

FECHA: primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene las funciones y clases necesarias para el entrenamiento de los distintos
             modelos de redes neuronales que se van a probar en el proyecto de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "BUILD_LSTM_MODEL"
    -FUNCION "ENTRENAR_LSTM_KFOLD"
    -FUNCION "ENTRENAR_TRANSFORMER_KFOLD"
    -FUNCION "CALCULAR_RESULTADOS_KFOLD"
    -CLASE   "POSITIONALENCONDING"
    -FUNCION "TRANSFORMER_ENCONDER"
    -FUNCION "BUILD_KERAS_TRANSFORMER_MODEL"
    -CLASE   "PLOTLEARNING"    
    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import tensorflow as tf
from tensorflow import keras
import numpy as np
import gc
import time
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from IPython.display import clear_output
from src import calcular_tiempo
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense

#####---------------------FIN LIBRERIAS-------------------------------------------------------------######






#####---------------------FUNCIONES Y CLASES--------------------------------------------------------###### 

###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------------FUNCION BUILD_LSTM_MODEL------------------------------------------------------
###   Función que construye y devuelve un modelo "LSTM" de clasificación, usando los hiperparámetros que recibe como argumento
###   ------------------------------------------------------------------------------------------------------------------------
    
def build_LSTM_model(num_capas_lstm,input_shape,hidden_state,mlp_units,lstm_dropout,mlp_dropout,n_classes):
     
    model = Sequential()
    
    model.add(LSTM(hidden_state, return_sequences=True, input_shape=(input_shape)))
    
    ## Añadimos tantas capas LSTM extra como hayamos recibido
    for _ in range(num_capas_lstm - 1):
        model.add(LSTM(hidden_state))
        
    model.add(Dropout(lstm_dropout))
    model.add(Dense(mlp_units, activation='relu'))
    model.add(Dropout(mlp_dropout))
    model.add(Dense(n_classes, activation='softmax'))
    
    return model
    

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   -------------------------------------FUNCION ENTRENAR_LSTM_KFOLD--------------------------------------------------------
###   Función que entrena un modelo "LSTM" de clasificación, haciendo uso de la técnica de KFold. Recibe el modelo como
###   argumento y lo devuelve con los pesos del mejor split del KFold. 
###   Cada split se entrena con los mismos pesos vírgenes que recibe como argumento.
###   Devuelve las métricas obtenidas durante el entrenamiento. 
###   ------------------------------------------------------------------------------------------------------------------------

def entrenar_LSTM_kfold(lista_KFold,X_train_bruto,y_train_bruto,dic_modelo,pesos_virgenes,model,num_capas_lstm,input_shape,hidden_state,mlp_units,lstm_dropout,mlp_dropout,learning_rate,batch_size,epochs,n_classes,rutaResultados,hashSemiflujos,hashModelo,guardar_resultados):
    
    
    best_val_accuracy = 0
    best_val_loss = 0
    lista_val_accuracy = []
    lista_val_loss = []
    lista_train_accuracy = []
    lista_train_loss = []
    
    ## Vamos a recorrer los splits del KFold, entrenando para cada split con los sets de entrenamiento
    ## y validación correspondientes
    for numSplit,indices_split in enumerate(lista_KFold):
        
        print("\n-------------COMENZANDO SPLIT "+str(numSplit)+"------------------\n")
        
        ## Reseteamos el modelo a los pesos vírgenes si no somos el primer split      
        if(numSplit != 0):   
            ## Borramos el modelo porque si no quedan restos del anterior en la RAM, saturándola en balde
            del model
            gc.collect()
            ## Construimos nuestro transformer de nuevo
            model = build_LSTM_model(num_capas_lstm,input_shape,hidden_state,mlp_units,lstm_dropout,mlp_dropout,n_classes)
            ## Lo compilamos eligiendo el tipo de loss, optimizador y métricas
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
            ## Ponemos los mismos pesos iniciales aleatorios del primer split
            model.set_weights(pesos_virgenes)
        
        ## Dentro de todo el "X_train_bruto" e "y_train_bruto" nos quedamos con aquellos semiflujos 
        ## que nos indica el índice correspondiente a este split. 
        ## indices_split[0] ---> contiene los índices de train   
        ## indices_split[1] ---> contiene los índices de validation 
        X_train = X_train_bruto[indices_split[0]]
        y_train = y_train_bruto[indices_split[0]]

        X_validation = X_train_bruto[indices_split[1]]
        y_validation = y_train_bruto[indices_split[1]]
        
        start_time = time.time()  
        ## Entrenamos y vamos mostrando la gráfica de accuracy y loss al final de cada época 
        ## En la última eṕoca se guarda la gráfica en local si así está seleccionado en el archivo de configuración
        history = model.fit(X_train,y_train,validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs,\
                  callbacks=[PlotLearning(rutaResultados,hashSemiflujos,hashModelo,guardar_resultados,epochs,numSplit)], verbose=1)     
        
        ## Guardamos el tiempo que ha tardado el entrenamiento de este split
        min_entrenamiento = int(calcular_tiempo(start_time,False))
        
        ## Guardamos el loss y el accuracy de training y validation de este split 
        lista_train_accuracy.append(history.history['accuracy'][-1])
        lista_train_loss.append(history.history['loss'][-1])   
        lista_val_accuracy.append(history.history['val_accuracy'][-1])
        lista_val_loss.append(history.history['val_loss'][-1])
        
        ## Si el accuracy supera al actual best_val_accuracy, sobreescribimos los datos con los de este split
        if (history.history['val_accuracy'][-1] > best_val_accuracy):
            best_val_accuracy = history.history['val_accuracy'][-1]
            best_val_loss = history.history['val_loss'][-1]
            best_weights = model.get_weights()
            ## X_validation_best = X_validation  ## Descomentar ambas líneas si se quiere hacer  
            ## y_validation_best = y_validation  ## la comprobación que existe antes del return
            dic_modelo['min_entrenamiento'] = min_entrenamiento

    
    ## Calculamos los valores medios y la desviación del KFold
    media_train_accuracy,media_train_loss,media_val_accuracy,media_val_loss,desviacion_train_acurracy,\
    desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss = \
    calcular_resultados_kfold(lista_train_accuracy,lista_train_loss,lista_val_accuracy,\
                              lista_val_loss,best_val_accuracy,best_val_loss)
    
    ## Dejamos el modelo con los pesos del mejor split
    model.set_weights(best_weights)
    
    
    ## Si se quiere comprobar que los pesos con los que hemos cargado el modelo coinciden con los resultados del mejor, 
    ## y no ha habido error, descomentar la siguiente línea de código y comprobar empíricamente como tiene la misma accuracy
    ## y loss que la información impresa en las líneas superiores para el mejor split.
    ## model.evaluate(X_validation_best, y_validation_best, batch_size=batch_size, verbose=1)
    
    
    return media_train_accuracy,media_train_loss,media_val_accuracy,media_val_loss,desviacion_train_acurracy,desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss,dic_modelo['min_entrenamiento']

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   -------------------------------------FUNCION ENTRENAR_TRANSFORMER_KFOLD-------------------------------------------------
###   Función que entrena un modelo "Transformer" de clasificación, haciendo uso de la técnica de KFold. Recibe el modelo como
###   argumento y lo devuelve con los pesos del mejor split del KFold. 
###   Cada split se entrena con los mismos pesos vírgenes que recibe como argumento.
###   Devuelve las métricas obtenidas durante el entrenamiento. 
###   ------------------------------------------------------------------------------------------------------------------------

def entrenar_transformer_kfold(lista_KFold,X_train_bruto,y_train_bruto,dic_modelo,pesos_virgenes,model,input_shape,n_classes,head_size,num_heads,positional_encoding,ff_dim,mlp_units,mlp_dropout,dropout,num_capas_atencion,learning_rate,batch_size,epochs,rutaResultados,hashSemiflujos,hashModelo,guardar_resultados):
    
    best_val_accuracy = 0
    best_val_loss = 0
    lista_val_accuracy = []
    lista_val_loss = []
    lista_train_accuracy = []
    lista_train_loss = []
    
    ## Vamos a recorrer los splits del KFold, entrenando para cada split con los sets de entrenamiento
    ## y validación correspondientes
    for numSplit,indices_split in enumerate(lista_KFold):
        
        print("\n-------------COMENZANDO SPLIT "+str(numSplit)+"------------------\n")
        
        ## Reseteamos el modelo a los pesos vírgenes si no somos el primer split      
        if(numSplit != 0):   
            ## Borramos el modelo porque si no quedan restos del anterior en la RAM, saturándola en balde
            del model
            gc.collect()
            ## Construimos nuestro transformer de nuevo
            model = build_keras_transformer_model(input_shape, n_classes,head_size, num_heads,ff_dim, mlp_units,\
                                      positional_encoding,dropout,mlp_dropout,num_capas_atencion)
            ## Lo compilamos eligiendo el tipo de loss, optimizador y métricas
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
            ## Ponemos los mismos pesos iniciales aleatorios del primer split
            model.set_weights(pesos_virgenes)
        
        ## Dentro de todo el "X_train_bruto" e "y_train_bruto" nos quedamos con aquellos semiflujos 
        ## que nos indica el índice correspondiente a este split. 
        ## indices_split[0] ---> contiene los índices de train   
        ## indices_split[1] ---> contiene los índices de validation 
        X_train = X_train_bruto[indices_split[0]]
        y_train = y_train_bruto[indices_split[0]]

        X_validation = X_train_bruto[indices_split[1]]
        y_validation = y_train_bruto[indices_split[1]]
        
        start_time = time.time()  
        ## Entrenamos y vamos mostrando la gráfica de accuracy y loss al final de cada época 
        ## En la última eṕoca se guarda la gráfica en local si así está seleccionado en el archivo de configuración
        history = model.fit(X_train,y_train,validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs,\
                  callbacks=[PlotLearning(rutaResultados,hashSemiflujos,hashModelo,guardar_resultados,epochs,numSplit)], verbose=1)     
        
        ## Guardamos el tiempo que ha tardado el entrenamiento de este split
        min_entrenamiento = int(calcular_tiempo(start_time,False))
        
        ## Guardamos el loss y el accuracy de training y validation de este split 
        lista_train_accuracy.append(history.history['accuracy'][-1])
        lista_train_loss.append(history.history['loss'][-1])   
        lista_val_accuracy.append(history.history['val_accuracy'][-1])
        lista_val_loss.append(history.history['val_loss'][-1])
        
        ## Si el accuracy supera al actual best_val_accuracy, sobreescribimos los datos con los de este split
        if (history.history['val_accuracy'][-1] > best_val_accuracy):
            best_val_accuracy = history.history['val_accuracy'][-1]
            best_val_loss = history.history['val_loss'][-1]
            best_weights = model.get_weights()
            ## X_validation_best = X_validation  ## Descomentar ambas líneas si se quiere hacer  
            ## y_validation_best = y_validation  ## la comprobación que existe antes del return
            dic_modelo['min_entrenamiento'] = min_entrenamiento

    ## Calculamos los valores medios y la desviación del KFold
    media_train_accuracy,media_train_loss,media_val_accuracy,media_val_loss,desviacion_train_acurracy,\
    desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss = \
    calcular_resultados_kfold(lista_train_accuracy,lista_train_loss,lista_val_accuracy,\
                              lista_val_loss,best_val_accuracy,best_val_loss)
    
    ## Dejamos el modelo con los pesos del mejor split
    model.set_weights(best_weights)
    
    
    ## Si se quiere comprobar que los pesos con los que hemos cargado el modelo coinciden con los resultados del mejor, 
    ## y no ha habido error, descomentar la siguiente línea de código y comprobar empíricamente como tiene la misma accuracy
    ## y loss que la información impresa en las líneas superiores para el mejor split.
    ## model.evaluate(X_validation_best, y_validation_best, batch_size=batch_size, verbose=1)
    
    
    return media_train_accuracy,media_train_loss,media_val_accuracy,media_val_loss,desviacion_train_acurracy,desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss,dic_modelo['min_entrenamiento']

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   -------------------------------------FUNCION CALCULAR_RESULTADOS_KFOLD------------------------------------------------
###   Función que calcula los valores medios y la desviación típica del KFold. (Train accuracy, train loss, val accuracy,
###   val loss.)
###   ------------------------------------------------------------------------------------------------------------------------

def calcular_resultados_kfold(lista_train_accuracy,lista_train_loss,lista_val_accuracy,lista_val_loss,best_val_accuracy,best_val_loss):
    
    ## Una vez terminado el KFold calculamos los valores medios
    media_train_accuracy = round (sum(lista_train_accuracy) / len(lista_train_accuracy),5)
    media_train_loss = round (sum(lista_train_loss) / len(lista_train_loss),5)
    media_val_accuracy = round (sum(lista_val_accuracy) / len(lista_val_accuracy),5)
    media_val_loss = round (sum(lista_val_loss) / len(lista_val_loss),5)
    
    ## Vamos a calcular la desviación estándar
    desviacion_train_acurracy = round(np.std(lista_train_accuracy),5)
    desviacion_train_loss = round(np.std(lista_train_loss),5)
    desviacion_val_acurracy = round(np.std(lista_val_accuracy),5)
    desviacion_val_loss = round(np.std(lista_val_loss),5)
    
    ## Imprimimos por pantalla toda esta información
    print("\n\nResultados finales del KFold:")
    print("\nLista con los training accuracy finales de cada split:")
    print(lista_train_accuracy)
    print("\nLista con los training loss finales de cada split:")
    print(lista_train_loss)
    print("\nLista con los validation accuracy finales de cada split:")
    print(lista_val_accuracy)
    print("\nLista con los validation loss finales de cada split:")
    print(lista_val_loss)
    print("\nMedia de train_accuracy de los diferentes splits = "+str(media_train_accuracy))
    print("Media de train_loss de los diferentes splits  = "+str(media_train_loss))
    print("\nMedia de val_accuracy de los diferentes splits = "+str(media_val_accuracy))
    print("Media de val_loss de los diferentes splits  = "+str(media_val_loss))
    
    print("\nDesviacion estandar del train_accuracy de los diferentes splits = "+str(desviacion_train_acurracy))
    print("Desviacion estandar del train_loss de los diferentes splits = "+str(desviacion_train_loss))
    print("\nDesviacion estandar del val_accuracy de los diferentes splits = "+str(desviacion_val_acurracy))
    print("Desviacion estandar del val_loss de los diferentes splits = "+str(desviacion_val_loss))
    
    print("\nBest val_accuracy = "+str(round(best_val_accuracy,5)))
    print("Loss del modelo con el mejor val_accuracy = "+str(round(best_val_loss,5)))
    
    return media_train_accuracy,media_train_loss,media_val_accuracy,media_val_loss,desviacion_train_acurracy,desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss
    
    
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------CLASE POSITIONALENCONDING------------------------------------------------------
###   Clase para crear el positional encoding de un modelo "Transformer".
###   Autoría de TensorFlow y Silvia (ex-becaria CEIEC): 
###   https://www.tensorflow.org/text/tutorials/transformer?hl=es-419#positional_encoding
###   ------------------------------------------------------------------------------------------------------------------------

## CORRECCIÓN (revisión IJIS, nov 2026): la versión anterior (a) devolvía la entrada sin sumar nada en "call",
## por lo que positional_encoding=True no tenía ningún efecto, y (b) copiaba la columna 0 en todas las dimensiones,
## de modo que, aunque se hubiera sumado, todas las dimensiones recibían el mismo valor. Además usaba unas
## dimensiones fijas (MAX_TOKENS=101, d_model=100) que no coinciden con la entrada (20 paquetes x 201 tokens).
## Esta versión calcula el encoding sinusoidal estándar (Vaswani et al., 2017) a partir de la forma real de la
## entrada: una posición por paquete (eje 1) y d_model = número de tokens por paquete (eje 2).
## El parámetro "escala" permite reducir su amplitud, ya que los tokens están normalizados en [0, 1].
## Con positional_encoding=False el modelo es idéntico al original (mismos pesos), por lo que los pesos del
## artículo se siguen cargando sin cambios.

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, escala=1.0, name=None, **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.escala = float(escala)

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # sin en índices pares (2i)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) # cos en índices impares (2i+1)
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build(self, input_shape):
        num_posiciones = int(input_shape[1])
        d_model = int(input_shape[2])
        self.pos_encoding = self.positional_encoding(num_posiciones, d_model) * self.escala
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"escala": self.escala})
        return config

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN CLASE-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------

  

###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION TRANSFORMER_ENCONDER---------------------------------------------------
###   Función que construye la capa encoder de un "Transformer".
###   Autoría de Keras: 
###   https://keras.io/examples/timeseries/timeseries_transformer_classification/
###   ------------------------------------------------------------------------------------------------------------------------

## NOTA (revisión IJIS, nov 2026): en la versión original la primera LayerNormalization se calculaba pero su
## salida no se usaba (la atención recibía "inputs"), por lo que Keras la elimina del grafo. El modelo del
## artículo tiene, por tanto, una única LayerNormalization (402 parámetros), tras la conexión residual de la
## atención. Se mantiene ese comportamiento por defecto (norm_previa=False) para poder cargar los pesos del
## artículo; norm_previa=True aplica la normalización previa a la atención (pre-norm), como en el ejemplo de Keras.

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, norm_previa=False):
    # Normalization and Attention
    if norm_previa:
        x_norm = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x_norm, x_norm)
    else:
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------FUNCION BUILD_KERAS_TRANSFORMER_MODEL---------------------------------------------------
###   Función que construye un modelo "Transformer" sencillo de clasificación. 
###   Autoría de Keras: 
###   https://keras.io/examples/timeseries/timeseries_transformer_classification/
###   ------------------------------------------------------------------------------------------------------------------------
    
def build_keras_transformer_model(input_shape, n_classes, head_size, num_heads, ff_dim, mlp_units, PE, dropout, mlp_dropout, num_capas_atencion, escala_PE=1.0, norm_previa=False):
    
    inputs = keras.Input(shape=input_shape)

    if PE == True:
        position = PositionalEncoding(escala=escala_PE)
        x = position(inputs)
    else:
        x = inputs
    
    
    for _ in range(num_capas_atencion):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, norm_previa=norm_previa)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    

    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)

        
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs, name='categorical_PD_transformer')

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   ---------------------------------------------CLASE PLOTLEARNING---------------------------------------------------------
###   Clase del tipo "keras callback" para ir imprimiendo por pantalla el error de validación y de training, así como su 
###   correspondiente accuracy. 
###   En una versión posterior se ha añadido la opción de guardar el plot de la última época de entrenamiento en 
###   una ruta que recibe como argumento. (También se le puede pasar el n° de split dek KFold que corresponda, para poder
###   diferenciar entre las distintas curvas de entrenamiento que genera cada split.)
###   Argumentos:
###       - rutaResultados: ruta donde se guardará la captura del diagrama en formato PNG
###       - hashSemiflujos: hash que identifica a los semiflujos utilizados
###       - hashModelo: hash que identifica al modelo utilizado
###       - guardarResultados: True o False, para habilitar o no la realización de la captura en disco
###       - num_epochs: número de épocas del entrenamiento, para saber cuándo es la última
###       - numSplit: número de Split del KFold que estamos ejecutando, para distinguir este diagrama del resto de splits
###
###   (La versión inicial de este callback es de StackOverflow, no me pertenece su autoría.)
###   ------------------------------------------------------------------------------------------------------------------------

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def __init__(self,rutaResultados,hashSemiflujos,hashModelo,guardarResultados,num_epochs,numSplit):
        self.rutaResultados = rutaResultados
        self.hashSemiflujos = hashSemiflujos
        self.hashModelo = hashModelo
        self.guardarResultados = guardarResultados
        self.num_epochs = num_epochs
        self.numSplit = numSplit
    
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        
        if(epoch + 1 == self.num_epochs and self.guardarResultados == True):
            plt.savefig(self.rutaResultados+'Graficas de entrenamiento/HS_'+self.hashSemiflujos+'_HM_'+self.hashModelo+'/NS_'+str(self.numSplit)+'_GraficaEntrenamiento.png',facecolor='white',bbox_inches = 'tight',dpi=150)
            ## Ejemplo con n° split del KFold = 2, hashSemiflujos = 1234, hasModelo = 6789, modelo = LSTM, se guarda en:
            ## "Resultados/Resultados_LSTM/Graficas de entrenamiento/HS_1234_HM_6789/NS_2_GraficaEntrenamiento.png"
        
        plt.show()
        
            
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN CLASE-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------

#####---------------------FIN FUNCIONES Y CLASES-------------------------------------------------------------###### 