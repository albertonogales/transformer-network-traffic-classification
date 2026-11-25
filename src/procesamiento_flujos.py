#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "procesamiento_flujos.py".

AUTOR: Carlos Arranz Luque.

FECHA: Primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'procesamiento flujos', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "EVALUAR_SEMIFLUJOS_GUARDADOS" 
    -FUNCION "DESECHAR_SEMIFLUJOS_REPETIDOS"
    -FUNCION "PROCESAR_RAW"
    -FUNCION "OBTENERFLUJOS"
    -FUNCION "OBTENERSEMIFLUJOS"

"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import configparser
import os
import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical


#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 

###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------FUNCION EVALUAR_SEMIFLUJOS_GUARDADOS------------------------------------------------
###   Función que recibe el path a los semiflujos guardados, y genera un CSV con las características de los mismos,
###   identificándolos por su hash.
###   ------------------------------------------------------------------------------------------------------------------------

def evaluar_semiflujos_guardados(pathRelativoSemiflujos):
    
    hiperparametros_dict = {}  ## El diccionario que contendrá la información de cada carpeta de semiflujos
    config = configparser.RawConfigParser()  ## Un lector del archivo de configuración
    creado = False
    
    ## Comprobamos todos los elementos existentes en el directorio donde se guardan los semiflujos
    for it in os.scandir(pathRelativoSemiflujos):
        ## Si es un directorio, entramos y obtenemos las características de los semiflujos
        if it.is_dir():
            ## Leemos el archivo de configuración
            config.read(it.path + '/config.cfg')
            hiperparametros_dict["hashSemiflujo"] = it.name
            
            ## Actualizamos nuestro diccionario con los hiperparámetros que se usaron para la creación de los semiflujos
            hiperparametros_dict.update(dict(config.items('Hiperparametros')))
            
            file_pesos_modelo = open(it.path + '/diccionario_semiflujos.obj', 'rb')
            ## Cargamos el diccionario de los semiflujos para poder saber cuántos hay de cada tipo
            dic_semiflujos = pickle.load(file_pesos_modelo)
            file_pesos_modelo.close()
            
            contenido_categorias = ""
            ## Bucle guardando en un string el contenido
            for key in dic_semiflujos:
                contenido_categorias += key +":"+ str(dic_semiflujos[key]) + "  "
            
            ## Guardamos en el diccionario el string generado en el paso anterior
            hiperparametros_dict["num_semiflujos_por_categoria"] = contenido_categorias
            
            ## Al final obtenemos un diccionario con las siguientes claves:
            ## hashSemiflujo, categorias_a_procesar, min_paquetes_flujo, tiempo_maximo_semiflujo,
            ## num_paquetes_semiflujo, longitud_raw, num_hex_token, token_separador, num_semiflujos_por_categoria
       
            ## Si somos la primera carpeta de semiflujos que exploramos, generamos un dataframe
            if (creado == False):
                df = pd.DataFrame.from_dict([hiperparametros_dict])
                creado = True
            ## De no ser los primeros, actualizamos el dataframe existente, añadiendo una nueva fila
            else:
                df.loc[len(df)]=hiperparametros_dict         
    
    ## Si hemos generado un dataframe, lo guardamos, sobreescribiendo el CSV anterior de existir
    if (creado == True):
        df.to_csv(pathRelativoSemiflujos+"caracteristicas_semiflujos.csv",index=False)
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------




###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------FUNCION DESECHAR SEMIFLUJOS REPETIDOS------------------------------------------------
###   Función que recibe un array de semiflujos; el diccionario de contenido del mismo, que indica cuántos semiflujos hay de 
###   cada categoría; y el número de clases existentes en ambos. 
###   La función crea un array de semiflujos nuevo, donde solo añade aquellos que son únicos. También crea el correspondiente 
###   array de etiquetas para el array de semiflujos.
###   La función también actualiza el diccionario de contenido que recibe para que se actualice con los datos únicos reales.
###   Por último devuelve ambos arrays. 
###   ------------------------------------------------------------------------------------------------------------------------

def desechar_semiflujos_repetidos(lista_semiflujos, Semiflujos_diccionario_contenido, num_clases_existentes):

    ## Vamos a ver cuántos semiflujos repetidos hay y eliminarlos

    comienzo = 0
    final = 0

    ## Inicializamos el numpy array que contendrá nuestros semiflujos únicos 
    semiflujos_unicos = lista_semiflujos[0:0]

    lista_arrays_etiqueta = list()

    for etiqueta, key in enumerate(Semiflujos_diccionario_contenido):
        # Vemos cuantos semiflujos hay de esta categoria
        num_semiflujos_cat = Semiflujos_diccionario_contenido[key]

        # Actualizamos el índice del array donde finalizan los datos de la categoría actual
        final += num_semiflujos_cat

        # Obtenemos los datos únicos de esta categoría
        unicos_cat = np.unique(lista_semiflujos[comienzo:final] ,axis=0)
        num_semiflujos_cat_unicos = unicos_cat.shape[0]
                  
        # Añadimos estos semiflujos unicos al array
        semiflujos_unicos = np.append(semiflujos_unicos,unicos_cat, axis=0)

        # Calculamos cuántos repetidos había en esta categoría
        repetidos = num_semiflujos_cat - num_semiflujos_cat_unicos
        if(num_semiflujos_cat != 0):
            porcentaje = round(repetidos/num_semiflujos_cat*100 ,2)
        else:
            porcentaje = 0

        print("\nHay "+str(repetidos)+" de semiflujos repetidos de "+str(num_semiflujos_cat)+" totales en la categoria "+key)
        print("Esto representa el "+ str(porcentaje)+"% de la categoría")
        print("Hay "+str(num_semiflujos_cat_unicos)+" flujos unicos realmente")

        # Actualizamos dónde comienza el índice de la siguiente categoría
        comienzo += num_semiflujos_cat

        # Generamos las etiquetas de la categoría actual teniendo en cuenta los datos únicos que hemos obtenido
        lista_arrays_etiqueta = lista_arrays_etiqueta + [etiqueta]*num_semiflujos_cat_unicos

        # Actualizamos el diccionario con los semiflujos únicos reales
        Semiflujos_diccionario_contenido[key] = num_semiflujos_cat_unicos


    # Transformamos la lista de etiquetas en una lista categórica    
    lista_arrays_etiqueta_categorical = to_categorical (lista_arrays_etiqueta, num_classes = num_clases_existentes)    

    # Imprimimos cómo quedan las dimensiones del array de semiflujos únicos y sus correspondientes etiquetas
    print("\nVamos a ver como se quedan las dimensiones de los datos después de quitar los repetidos:")
    print("Array semiflujos: "+str(semiflujos_unicos.shape))
    print("Array etiquetas: "+str(lista_arrays_etiqueta_categorical.shape))
    print("Diccionario de contenido: "+str(Semiflujos_diccionario_contenido))
    
    return semiflujos_unicos,lista_arrays_etiqueta_categorical


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------FUNCION PROCESAR RAW-----------------------------------------------------------
###   Función que recibe el contenido RAW de un paquete, y un conjunto de hiperparámetros como la longitud, que define el 
###   número de caracteres hexadecimales que debe tener el paquete; el número de hexadecimales por token y el token separador.
###   Procesa el contenido del paquete y lo divide en subtokens, siendo el último el separador, dejando todo en un numpy array
###   que devuelve.
###   ------------------------------------------------------------------------------------------------------------------------

def procesar_RAW(payload_data, longitud_raw, num_hex_token, token_separador): 
    if( len(payload_data) < longitud_raw ):
        ## Hacemos padding de ceros si no llega a la longitud introducida como argumento de caracteres hexadecimales
        payload_data = payload_data.ljust(longitud_raw,"0")
    elif( len(payload_data) > longitud_raw ):
        ## Truncamos el payload a la longitud introducida como argumento de caracteres hexadecimales si 
        ## tiene un tamaño superior a este
        payload_data = payload_data[0:longitud_raw]
    
    ## Separamos el dato en paquetes de tantos hexadecimales como hayamos recibido como argumento
    payload_data = [payload_data[i:i+num_hex_token] for i in range(0, len(payload_data), num_hex_token)]
    
    ## Pasamos de hexadecimal a entero estos tokens
    payload_data = [int(x, 16) for x in payload_data]
    
    
    ## Optimizar el tipo de dato según se necesita más o menos tamaño, 4 caracteres hexadecimales tiene justo
    ## el tamaño de un entero de 16 bits, 8 caracteres hexadecimales tiene el tamaño de un entero de 32 bits.
    ## El máximo será 16 caracteres, ya que necesita un entero de 64 bits para no hacer overflow, que es el
    ## entero más grande disponible
    if(num_hex_token <= 4):
        tipo_dato = np.uint16 
    elif(num_hex_token <= 8):
        tipo_dato = np.uint32
    else:
        tipo_dato = np.uint64
    
    ## De momento solo se ha implementado la estrategia de número máximo como token separador
    if (token_separador == 'num_maximo'):
        ## Hacemos que el token separador sea el mayor número posible del tipo de dato que tengamos
        payload_data.append(np.iinfo(tipo_dato).max)
    
    ## Transformamos este payload tokenizado a np array
    array_transformado = np.array(payload_data,dtype=tipo_dato)
        
    
    return array_transformado

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------FUNCION OBTENERFLUJOS---------------------------------------------------
###   Función que genera y devuelve una lista con los distintos flujos/conversaciones que existan dentro de un dataframe
###   que recibe como argumento. Este dataframe es ya un dataframe semiprocesado de un archivo pcap.
###   La lista de flujos es una lista de dataframes, cada dataframe contiene el conjunto de paquetes que pertenecen al mismo
###   flujo.
###   ------------------------------------------------------------------------------------------------------------------------

def obtenerFlujos(df, min_paquetes_flujo): 
    
 
    ## Comprobamos cuantos flujos hay, cada hash unico de nuestro DF identifica uno
    lista_hashes = df.Hash.unique()

    ## Lista donde guardaremos cada uno de los flujos del dataframe
    lista_flujos = []
   
    # Creamos un dataframe para cada conversación/flujo de nuestro dataframe original
    for hash_flujo in lista_hashes:
        df_flujo = df[df["Hash"] == hash_flujo]
        
        ## Solo hacemos append a la lista si cumple el mínimo de paquetes por flujo, si no
        ## es descartado ya que es un flujo tan irrisorio que no aporta nada
        if(df_flujo.shape[0] >= min_paquetes_flujo ):
            #Eliminamos la columna que tenga vacía, ya sea la TCPPayload o la UDPdata, y reseteamos el índice
            df_flujo.reset_index(drop=True, inplace = True)
            df_flujo = df_flujo.dropna(axis='columns')
            lista_flujos.append(df_flujo)
    
                                                    
    ## Devolvemos esta lista que contiene los flujos en formato dataframe        
    return lista_flujos

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------FUNCION OBTENERSEMIFLUJOS---------------------------------------------------
###   Función que añade a una lista de semiflujos que recibe como argumento, los distintos semiflujos que existan dentro de
###   una lista de flujos que recibe como argumento.
###   Vamos a recorrer los distintos paquetes de cada flujo y creando semiflujos que cumplan con los requisitos recibidos
###   como argumento, esto quiere decir obtener un semiflujo con paquetes que tengan entre sí una diferencia de tiempo
###   menor o igual al 'tiempo_maximo_semiflujo' y con un número de paquetes en total igual a 'num_paquetes_semiflujo',
###   si el semiflujo tiene menos paquetes se desecha.
###   ------------------------------------------------------------------------------------------------------------------------

def obtenerSemiflujos(lista_flujos, lista_arrays, token_separador, num_paquetes_semiflujo, tiempo_maximo_semiflujo, longitud_raw, num_hex_token): 
    
    ## Obtenemos cuantos flujos hay en nuestra lista para poder ir mostrando el avance por pantalla    
    num_flujos = len(lista_flujos)
        
    ## Vamos a recorrer los distintos flujos
    for cont,flujo in enumerate(lista_flujos):
        
        print("--------Comenzando el flujo numero "+str(cont+1)+" de "+str(num_flujos))
                
        ## Inicializamos el contador a cero  
        contador_paquetes_semiflujo = 0
        
        for i in range(len(flujo)):
            # flujo.iloc[i] me devuelve la fila i-ésima del flujo 
            # Con flujo.iloc[i][n] accedo al n-ésimo campo de la fila i-ésima del flujo
            # Solo tengo 3 campos por fila, tiempo, datos en RAW y hash, en ese orden
            
            if( (contador_paquetes_semiflujo==0) or ((flujo.iloc[i,0] - tiempo_inicio_semiflujo) > tiempo_maximo_semiflujo) ):
                ## Hay dos opciones:
                ## - Se acaba de terminar un semiflujo y voy a comenzar uno nuevo
                ## - Tengo un paquete que excede del limite de tiempo del semiflujo actual en construcción, 
                ##   se desecha el semiflujo y se empieza uno nuevo
            
                ## Reseteamos el semiflujo con este paquete como primero
                tiempo_inicio_semiflujo = flujo.iloc[i,0]
                ## Obtenemos el payload_data de este paquete
                payload_data = flujo.iloc[i,1]
                ## Creamos un numpy array tokenizando el payload_data del paquete
                array_nuevo_semiflujo = procesar_RAW(payload_data, longitud_raw, num_hex_token, token_separador)
                
                ## Inicializo un array bidimensional para el semiflujo, donde la primera dimensión es el n° de paquete
                ## y la segunda los distintos tokens en los que se ha dividido el payload_data del paquete
                array_semiflujo = np.array([array_nuevo_semiflujo])
                
                ## Actualizo el contador
                contador_paquetes_semiflujo = 1
                
            else:
                ## Estoy con un paquete que va a ser añadido al semiflujo
                contador_paquetes_semiflujo+=1

                ## Obtenemos el payload_data de este paquete
                payload_data = flujo.iloc[i,1]
                ## Creamos un numpy array tokenizando el payload_data del paquete
                array_nuevo = procesar_RAW(payload_data, longitud_raw, num_hex_token, token_separador)
                
                ## Añado al array bidimensional un nuevo paquete con su respectivos tokens
                array_semiflujo=np.append(array_semiflujo, [array_nuevo],axis = 0)

                
                ## Comprobamos si tenemos un semiflujo terminado 
                if (contador_paquetes_semiflujo == num_paquetes_semiflujo): 
                    ## Añado a la lista de arrays bidimensionales el que acabo de componer 
                    lista_arrays.append(array_semiflujo)
                    ## Reseteamos el contador de paquetes
                    contador_paquetes_semiflujo=0
                       
                                              
    ## Hemos terminado :D       
    
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------

#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 