#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
##### 
#####               ESTE ARCHIVO CONTIENE FUNCIONES QUE YA NO SE USAN, PERO QUE ERA UNA PENA  
#####               TIRAR AL ESPACIO DE UNOS Y CEROS OLVIDADOS QUE ES LA PAPELERA DE RECICLAJE
#####               
#####               CONTIENE IMPLEMENTACIONES PARALELIZADAS DE LAS FUNCIONES QUE ESTÁN DESCRITAS
#####               EN EL ARCHIVO 'procesamiento_flujos.py'
#####
#####               AL FINAL DEBIDO A LA ESCASA GANANCIA Y LA COMPLEJIDAD QUE PUEDE SUPONER MANTENER
#####               ESTE CÓDIGO POR OTRA PERSONA, SE HA DECIDIDO MANTENER LA IMPLEMENTACIÓN SECUENCIAL
#####
#####               DE MOMENTO NO ESTÁ NI TAN BIEN COMENTADO NI EXPLICADO COMO EL RESTO DE ARCHIVOS
#####
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####----------------------¡¡¡¡¡¡¡¡WARNING!!!!!!----------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "procesamiento_flujos_paralelizado.py".

AUTOR: Carlos Arranz Luque.

FECHA: Primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'procesamiento flujos', para el proyecto
             de clasificación de tráfico de red.

ÍNDICE DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "OBTENERFLUJOSPARALELIZADA"
    -FUNCION "OBTENERFLUJOSPARALELIZADAHIJA"
    -FUNCION "OBTENERSEMIFLUJOSPARALELIZADA"
    -FUNCION "OBTENERSEMIFLUJOSPARALELIZADAHIJA"
    -FUNCION "OBTENERFLUJOSCUJO"

"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import os
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
from shutil import which
import multiprocessing
from multiprocessing import Pool
import psutil
from multiprocessing import Manager
from src import procesar_RAW

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 

## Variable global necesaria
dataframe_global = 0

#####---------------------FUNCIONES-----------------------------------------------------------------###### 

###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION OBTENERFLUJOSPARELELIZAD-----------------------------------------------
###   Función que genera y devuelve una lista con los distintos flujos/conversaciones que existan dentro de un dataframe
###   que recibe como argumento. Este dataframe es ya un dataframe semiprocesado de un archivo pcap.
###   ------------------------------------------------------------------------------------------------------------------------

def obtenerFlujosParalelizada(df, min_paquetes_flujo):
    
    ## Obtenemos la cantidad de cores del pc (Al final se han fijado los cores manualmente para pruebas)
    #  n_cores = multiprocessing.cpu_count()  
    
    ## Comprobamos cuantos flujos hay, cada hash unico de nuestro DF identifica uno
    lista_hashes = df.Hash.unique()
    
    num_hashes_unicos = len(lista_hashes)
    
    if(num_hashes_unicos == 0):
        return []
    
    elif(num_hashes_unicos <= 4):
        n_cores = num_hashes_unicos
    else:
        n_cores = 4

    
    ## Dividimos la lista según estos cores
    lista_spliteada = np.array_split(lista_hashes, n_cores)
    
    lista_definitiva = []
    
    #for i in range(n_cores):
        #lista_definitiva.append((lista_spliteada[i],df,min_paquetes_flujo))
        
    global dataframe_global
    
    dataframe_global = df
        
    for i in range(n_cores):
        lista_definitiva.append((lista_spliteada[i],min_paquetes_flujo))

    with Pool(n_cores) as p:
        lista_flujos = p.starmap(obtenerFlujosParalelizadaHija,lista_definitiva)
    
    #aplanar la lista
    lista_flujos = [item for sublist in lista_flujos for item in sublist]
    
    return lista_flujos
    

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------FUNCION OBTENERFLUJOSPARELELIZADAHIJA---------------------------------------------
###   Función que genera y devuelve una lista con los distintos flujos/conversaciones que existan dentro de un dataframe
###   que recibe como argumento. Este dataframe es ya un dataframe semiprocesado de un archivo pcap.
###   ------------------------------------------------------------------------------------------------------------------------
    
#def obtenerFlujosParalelizadaHija(lista_hashes, df, min_paquetes_flujo):    
def obtenerFlujosParalelizadaHija(lista_hashes, min_paquetes_flujo):
    
    global dataframe_global
    
    df = dataframe_global
    
    ## Lista donde guardaremos cada uno de los flujos del dataframe
    lista_flujos = []
    
    
    if (len(lista_hashes) != 0):
        for hash0 in lista_hashes:
            df_flujo = df[df["Hash"] == hash0]

            ## Solo hacemos append a la lista si cumple el mínimo de paquetes por flujo, si no
            ## es descartado ya que es un flujo tan irrisorio que no aporta nada
            if(df_flujo.shape[0] >= min_paquetes_flujo ):
                #Eliminamos la columna que tenga vacía, ya sea la TCPPayload o la UDPdata
                df_flujo.reset_index(drop=True, inplace = True)
                df_flujo = df_flujo.dropna(axis='columns')
                lista_flujos.append(df_flujo) 



        # Para cada dataframe de la lista de flujos que acabamos de crear:
        #    -Eliminamos la columna de datos RAW que sobre, ya sea la de TCPpayload o la de UDPdata, la que tenga vacía
        #    -También reseteamos el índice

        #for contador,df in enumerate(lista_flujos):
            #lista_flujos[contador]=df.dropna(axis='columns')
            #lista_flujos[contador].reset_index(drop=True, inplace = True)
    
                                                    
    ## Devolvemos esta lista que contiene los flujos en formato dataframe        
    return lista_flujos


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------FUNCION OBTENERSEMIFLUJOSPARELELIZADA-------------------------------------------
###   Función que añade a una lista de semiflujos que recibe como argumento, los distintos semiflujos que existan dentro de
###   una lista de flujos que recibe como argumento
###   ------------------------------------------------------------------------------------------------------------------------

def obtenerSemiflujosParalelizada(lista_flujos, lista_arrays, token_separador, max_paquetes_semiflujo, tiempo_maximo_semiflujo, longitud_raw, num_hex_token): 
    
    ## Obtenemos la cantidad de cores del pc
    n_cores = multiprocessing.cpu_count()
    
    num_flujos = len(lista_flujos)
    
    if(num_flujos == 0):
        return
    
    if (num_flujos <= 8 ):
        n_cores = num_flujos
    else:
        n_cores = 8
    

    ## Dividimos la lista según estos cores
    
    lista_spliteada = np.array_split(lista_flujos, n_cores)
    
    

    lista_definitiva = []


    for i in range(n_cores):
        lista_definitiva.append((lista_spliteada[i],token_separador,max_paquetes_semiflujo,tiempo_maximo_semiflujo, longitud_raw, num_hex_token))

    with Pool(n_cores) as p:
        #lista_semiflujos_hijos
        lista_arrays_hijos = p.starmap(obtenerSemiflujosParalelizadaHija,lista_definitiva)


    #if (len(lista_semiflujos_hijos) != 0):
        #lista_semiflujos_hijos = [item for sublist in lista_semiflujos_hijos for item in sublist]

    
    for lista in lista_arrays_hijos:
        if (lista is not None):
            for item in lista:
                lista_arrays.append(item)
        
    #lista_arrays_hijos = [item for sublist in lista_arrays_hijos for item in sublist]
    #return lista_semiflujos_hijos
    #return lista_arrays_hijos

    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------FUNCION OBTENERSEMIFLUJOSPARELELIZADAHIJA--------------------------------------------
###   Función que añade a una lista de semiflujos que recibe como argumento, los distintos semiflujos que existan dentro de
###   una lista de flujos que recibe como argumento
###   ------------------------------------------------------------------------------------------------------------------------
    

def obtenerSemiflujosParalelizadaHija(lista_flujos, token_separador, max_paquetes_semiflujo, tiempo_maximo_semiflujo, longitud_raw, num_hex_token):


    lista_arrays = []
    
    
    if(len(lista_flujos) != 0):
    
        ## Obtenemos cuantos flujos hay en nuestra lista para poder ir mostrando el avance por pantalla    
        num_flujos = len(lista_flujos)

        ## Vamos a recorrer los distintos flujos
        for cont,flujo in enumerate(lista_flujos):
            
            ###----------------------NUEVO-----------------------------
            flujo = pd.DataFrame(flujo)
            ###----------------------NUEVO-----------------------------
            
            print("--------Comenzando el flujo numero "+str(cont+1)+" de "+str(num_flujos))

            ## Inicializamos el contador a cero  
            contador_paquetes_semiflujo = 0

            ## Vamos a recorrer los distintos paquetes del flujo y creando semiflujos que cumplan con los requisitos recibidos
            ## como argumento, esto quiere decir obtener un semiflujo con paquetes que tengan entre sí una diferencia de tiempo
            ## menor o igual al 'tiempo_maximo_semiflujo' y con un número de paquetes en total igual a 'max_paquetes_semiflujo',
            ## si el semiflujo tiene menos paquetes se desecha.
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
                    ### ---------------------NUEVO---------------------------
                    ## Obtenemos el payload_data de este paquete
                    payload_data = flujo.iloc[i,1]
                    ## Creamos un numpy array tokenizando el payload_data del paquete
                    array_nuevo_semiflujo = procesar_RAW(payload_data, longitud_raw, num_hex_token, token_separador)

                    ## Inicializo un array bidimensional para el semiflujo, donde la primera dimensión es el n° de paquete
                    ## y la segunda los distintos tokens en los que se ha dividido el payload_data del paquete
                    array_semiflujo = np.array([array_nuevo_semiflujo])
                    ### ---------------------FIN NUEVO---------------------------
                    ## Actualizo el contador
                    contador_paquetes_semiflujo = 1

                else:
                    ## Estoy con un paquete que va a ser añadido al semiflujo
                    contador_paquetes_semiflujo+=1
                    ## --------------------
                    ### ---------------------NUEVO--------------------------- 
                    ## Obtenemos el payload_data de este paquete
                    payload_data = flujo.iloc[i,1]
                    ## Creamos un numpy array tokenizando el payload_data del paquete
                    array_nuevo = procesar_RAW(payload_data, longitud_raw, num_hex_token, token_separador)

                    ## Añado al array bidimensional un nuevo paquete con su respectivos tokens
                    array_semiflujo=np.append(array_semiflujo, [array_nuevo],axis = 0)
                    ### ---------------------FIN NUEVO---------------------------

                    ## Comprobamos si tenemos un semiflujo terminado 
                    if (contador_paquetes_semiflujo == max_paquetes_semiflujo):
                        ### ---------------------NUEVO--------------------------- 
                        ## Añado a la lista de arrays bidimensionales el que acabo de componer 
                        lista_arrays.append(array_semiflujo)
                        ### ---------------------FIN NUEVO---------------------------
                        ## Reseteamos el contador de paquetes
                        contador_paquetes_semiflujo=0
                        
                        
        return lista_arrays
                        
                 
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION OBTENERFLUJOSCUJO-----------------------------------------------
###   Función que genera y devuelve una lista con los distintos flujos/conversaciones que existan dentro de un dataframe
###   que recibe como argumento. Este dataframe es ya un dataframe semiprocesado de un archivo pcap.
###   En honor a J.E.C.B por su inestimable ayuda :3
###   ------------------------------------------------------------------------------------------------------------------------

def obtenerFlujosCUJO(df, min_paquetes_flujo):
    lista_hashes = df.Hash.unique()
    
    df_flujos = df[df.groupby("Hash")["Hash"].transform("size") >= min_paquetes_flujo].dropna(axis=1).reset_index(drop=True)
    
    return [df_flujos[df_flujos["Hash"] == hash] for hash in lista_hashes if not df_flujos[df_flujos["Hash"] == hash].empty]

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------
