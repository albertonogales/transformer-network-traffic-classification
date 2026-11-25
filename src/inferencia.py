#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "inferencia.py".

AUTOR: Carlos Arranz Luque.

FECHA: primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'inferencia', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "OBTENER_HIPERPARAMETROS_XFMR"
    -FUNCION "OBTENER_HIPERPARAMETROS_LSTM"
    -FUNCION "RECONSTRUIR_LSTM"
    -FUNCION "RECONSTRUIR_XFMR"
    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import configparser
import numpy as np
import pandas as pd
from math import ceil
from src import build_LSTM_model,build_keras_transformer_model,obtener_pesos_modelo
from tensorflow import keras

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 


###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------FUNCION OBTENER_HIPERPARAMETROS_XFMR------------------------------------------------
###   Función que dado unas rutas y hashes de un modelo XFMR entrenado previamente con unos semiflujos, obtiene todos los 
###   hiperparámetros y datos necesarios para recrear el modelo, devolviéndolos en el return final.
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_hiperparametros_XFMR(rutaConfigXFMR,hash_pesos_modelo,hash_pesos_semiflujos,rutaSemiflujos):
    
    ## Leemos el archivo ".cfg" donde están las opciones de ejecución e hiperparámetros 
    ## del modelo XFMR que queremos recuperar
    config = configparser.RawConfigParser()
    config.read(rutaConfigXFMR+hash_pesos_modelo+'.cfg')

    ## Optenemos los valores de los hiperparámetros usados para crear el modelo
    hiperparametros_dict = dict(config.items('Hiperparametros'))
    
    batch_size = int(hiperparametros_dict['batch_size'])
    learning_rate = float(hiperparametros_dict['learning_rate'])
    positional_encoding = eval(hiperparametros_dict['positional_encoding'])
    num_heads = int(hiperparametros_dict['num_heads'])
    head_size = int(hiperparametros_dict['head_size'])
    mlp_units = int(hiperparametros_dict['mlp_units'])
    ff_dim = int(hiperparametros_dict['ff_dim'])
    mlp_dropout = float(hiperparametros_dict['mlp_dropout'])
    dropout = float(hiperparametros_dict['dropout'])
    num_capas_atencion = int(hiperparametros_dict['num_capas_atencion'])
    normalizar = eval(hiperparametros_dict['normalizar'])
    
    ## Leemos el CSV de los semiflujos existentes, para encontrar las características que necesitamos
    ## de los semiflujos que se usaron para entrenar el modelo
    csv_semiflujos = pd.read_csv(rutaSemiflujos+'caracteristicas_semiflujos.csv') 
    
    ## Nos quedamos con la fila que nos da la información de los semiflujos usados
    datos_semiflujos = csv_semiflujos.loc[csv_semiflujos['hashSemiflujo'] == hash_pesos_semiflujos]
    
    if(datos_semiflujos.empty == True):
        print("\n\nERROR!!\nHash de semiflujos no encontrado en el CSV de caracteristicas de semiflujos!!")
        return
    
    ## Obtenemos el número de clases
    n_clases = len(datos_semiflujos["categorias_a_procesar"].iloc[0].split())
    
    ## Obtenemos las clases 
    clases_clasificador = datos_semiflujos["categorias_a_procesar"].iloc[0].split()
    
    ## Obtenemos el input shape (num_paquetes_semiflujo, num_tokens_por_paquete)
    input_shape = (datos_semiflujos["num_paquetes_semiflujo"].iloc[0], ceil(datos_semiflujos["longitud_raw"].iloc[0]/datos_semiflujos["num_hex_token"].iloc[0])+1)
    
    ## Devolvemos todos estos datos obtenidos
    return batch_size,learning_rate,positional_encoding,num_heads,head_size,mlp_units,ff_dim,mlp_dropout,dropout,\
num_capas_atencion,normalizar,n_clases,clases_clasificador,input_shape
    


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------FUNCION OBTENER_HIPERPARAMETROS_LSTM------------------------------------------------
###   Función que dado unas rutas y hashes de un modelo LSTM entrenado previamente con unos semiflujos, obtiene todos
###   los hiperparámetros y datos necesarios para recrear el modelo, devolviéndolos en el return final.
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_hiperparametros_LSTM(rutaConfigLSTM,hash_pesos_modelo,hash_pesos_semiflujos,rutaSemiflujos):
    
    ## Leemos el archivo ".cfg" donde están las opciones de ejecución e hiperparámetros 
    ## del modelo LSTM que queremos recuperar
    config = configparser.RawConfigParser()
    config.read(rutaConfigLSTM+hash_pesos_modelo+'.cfg')

    ## Optenemos los valores de los hiperparámetros usados para crear el modelo
    hiperparametros_dict = dict(config.items('Hiperparametros'))

    num_capas_lstm = int(hiperparametros_dict['num_capas_lstm'])
    hidden_state = int(hiperparametros_dict['hidden_state'])
    mlp_units = int(hiperparametros_dict['mlp_units'])
    lstm_dropout = float(hiperparametros_dict['lstm_dropout'])
    mlp_dropout = float(hiperparametros_dict['mlp_dropout'])
    batch_size = int(hiperparametros_dict['batch_size'])
    learning_rate = float(hiperparametros_dict['learning_rate'])
    normalizar = eval(hiperparametros_dict['normalizar'])
    
    ## Leemos el CSV de los semiflujos existentes, para encontrar las características que necesitamos
    ## de los semiflujos que se usaron para entrenar el modelo
    csv_semiflujos = pd.read_csv(rutaSemiflujos+'caracteristicas_semiflujos.csv') 
    
    ## Nos quedamos con la fila que nos da la información de los semiflujos usados
    datos_semiflujos = csv_semiflujos.loc[csv_semiflujos['hashSemiflujo'] == hash_pesos_semiflujos]
    
    if(datos_semiflujos.empty == True):
        print("\n\nERROR!!\nHash de semiflujos no encontrado en el CSV de caracteristicas de semiflujos!!")
        return
    
    ## Obtenemos el número de clases
    n_clases = len(datos_semiflujos["categorias_a_procesar"].iloc[0].split())
    
    ## Obtenemos las clases 
    clases_clasificador = datos_semiflujos["categorias_a_procesar"].iloc[0].split()
    
    ## Obtenemos el input shape (num_paquetes_semiflujo, num_tokens_por_paquete)
    input_shape = (datos_semiflujos["num_paquetes_semiflujo"].iloc[0], ceil(datos_semiflujos["longitud_raw"].iloc[0]/datos_semiflujos["num_hex_token"].iloc[0])+1)
    
    ## Devolvemos todos estos datos obtenidos
    return num_capas_lstm,hidden_state,mlp_units,lstm_dropout,mlp_dropout,batch_size,learning_rate,\
normalizar,n_clases,clases_clasificador,input_shape
    


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------------FUNCION RECONSTRUIR_LSTM------------------------------------------------------
###   Función que dado unos hiperparámetros y opciones de un modelo LSTM, junto con sus correspondientes pesos, lo devuelve
###   correctamente creado
###   ------------------------------------------------------------------------------------------------------------------------

def reconstruir_LSTM(num_capas_lstm,hidden_state,mlp_units,lstm_dropout,mlp_dropout,learning_rate,\
                     n_clases,input_shape,hash_pesos_semiflujos,hash_pesos_modelo,rutaPesosLSTM):
    
    ## Construimos nuestro transformer clasificador
    model = build_LSTM_model(num_capas_lstm,input_shape,hidden_state,mlp_units,lstm_dropout,mlp_dropout,n_clases)
    
    ## Lo compilamos eligiendo el tipo de loss, optimizador y métricas
    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
    
    ## Obtenemos los pesos que nos han indicado cargar
    dic_modelo = obtener_pesos_modelo(hash_pesos_semiflujos, hash_pesos_modelo, rutaPesosLSTM, 'LSTM')
    ## Cargamos los pesos entrenados al modelo
    model.set_weights(dic_modelo['pesos'])
    
    return model

    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------FUNCION RECONSTRUIR_XFMR--------------------------------------------------------
###   Función que dado unos hiperparámetros y opciones de un modelo Transformer, junto con sus correspondientes pesos,
###   lo devuelve correctamente creado
###   ------------------------------------------------------------------------------------------------------------------------

def reconstruir_XFMR(n_clases,learning_rate,positional_encoding,num_heads,head_size,mlp_units,ff_dim,mlp_dropout,dropout,num_capas_atencion,input_shape,hash_pesos_semiflujos,hash_pesos_modelo,rutaPesosLSTM):
      
    ## Construimos nuestro transformer clasificador
    model = build_keras_transformer_model(input_shape, n_clases,head_size, num_heads,ff_dim, mlp_units,\
                                          positional_encoding,dropout,mlp_dropout,num_capas_atencion)

    ## Lo compilamos eligiendo el tipo de loss, optimizador y métricas
    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=["accuracy"])
    
    ## Obtenemos los pesos que nos han indicado cargar
    dic_modelo = obtener_pesos_modelo(hash_pesos_semiflujos, hash_pesos_modelo, rutaPesosLSTM, 'XFMR')
    ## Cargamos los pesos entrenados al modelo
    model.set_weights(dic_modelo['pesos'])
    
    return model

    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 
