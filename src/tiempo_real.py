#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "tiempo_real.py".

AUTOR: Carlos Arranz Luque.

FECHA: primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'tiempo real', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "OBTENER_PARAMETROS_PREPROCESAMIENTIO"
    -FUNCION "PCAP_A_SEMIFLUJOS"
    -FUNCION "CLASIFICAR_SEMIFLUJOS"
    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import pandas as pd
import numpy as np
from numpy import dstack
from sklearn.preprocessing import normalize
from collections import Counter
import matplotlib.pyplot as plt
from src import generar_CSVs
from src import generar_DFs
from src import obtenerFlujos
from src import obtenerSemiflujos
from IPython.display import clear_output

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------FUNCION OBTENER_PARAMETROS_PREPROCESAMIENTIO-----------------------------------------------
###   Función que dado un hash de semiflujos, devuelve los parametros utilizados para su preprocesamiento
###   Listado de parámetros a devolver:
###    -min_paquetes_flujo
###    -token_separador
###    -num_paquetes_semiflujo
###    -tiempo_maximo_semiflujo
###    -longitud_raw
###    -num_hex_token
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_parametros_preprocesamiento(rutaSemiflujos,hash_pesos_semiflujos):
    
    ## Leemos el CSV de los semiflujos existentes, para encontrar las características que necesitamos
    csv_semiflujos = pd.read_csv(rutaSemiflujos+'caracteristicas_semiflujos.csv') 

    ## Nos quedamos con la fila que nos da la información de los semiflujos que hemos recibido como argumento
    datos_semiflujos = csv_semiflujos.loc[csv_semiflujos['hashSemiflujo'] == hash_pesos_semiflujos]

    if(datos_semiflujos.empty == True):
        print("\n\nERROR!!\nHash de semiflujos no encontrado en el CSV de caracteristicas de semiflujos!!")
        return

    ## Obtenemos las características que necesitamos del preprocesamiento empleado
    min_paquetes_flujo = datos_semiflujos["min_paquetes_flujo"].iloc[0]

    token_separador = datos_semiflujos["token_separador"].iloc[0]

    num_paquetes_semiflujo = datos_semiflujos["num_paquetes_semiflujo"].iloc[0]

    tiempo_maximo_semiflujo = datos_semiflujos["tiempo_maximo_semiflujo"].iloc[0]

    longitud_raw = datos_semiflujos["longitud_raw"].iloc[0]

    num_hex_token = datos_semiflujos["num_hex_token"].iloc[0]

    return min_paquetes_flujo,token_separador,num_paquetes_semiflujo,tiempo_maximo_semiflujo,longitud_raw,num_hex_token

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------




###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------FUNCION PCAP_A_SEMIFLUJOS-----------------------------------------------------------
###   Función que dado un archivo PCAP recién capturado, realiza todos los pasos necesarios obtener sus semiflujos
###   y devolverlos
###   ------------------------------------------------------------------------------------------------------------------------

def PCAP_a_semiflujos(pathPCAPs,pathCSVs,nombrePCAPprocesar,normalizar,min_paquetes_flujo,token_separador,num_paquetes_semiflujo,tiempo_maximo_semiflujo,longitud_raw,num_hex_token):
    
    
    ## Generamos el CSV a partir del PCAP 
    generar_CSVs(pathPCAPs, pathCSVs,True, nombrePCAPprocesar)

    nombreCSVprocesar = nombrePCAPprocesar + ".csv"

    ## Procesamos el CSV y lo convertimos a DF con las características relevantes
    df = generar_DFs(pathCSVs, True, nombreCSVprocesar)

    ## Vamos a buscar los flujos y semiflujos del DF

    ## La lista que contendrá los semiflujos
    lista_semiflujos = list()

    ## Vamos a obtener una lista de flujos del dataframe
    lista_flujos = obtenerFlujos(df, min_paquetes_flujo)    
    print("----Numero de flujos obtenidos en el DF igual a "+str(len(lista_flujos))+"\n")

    ## Función que dada una lista de flujos, añade los semiflujos que encuentre a la lista de semiflujos
    obtenerSemiflujos(lista_flujos, lista_semiflujos, token_separador, num_paquetes_semiflujo, tiempo_maximo_semiflujo,longitud_raw,num_hex_token)
    
    
    ## Si hemos obtenido semiflujos los convertimos a numpy array
    if (len(lista_semiflujos)!=0):
        ## A continuación se deja la lista como un np array tridimensional, donde la primera dimensión es el número de semiflujos,
        ## la segunda es los paquetes que compone dicho semiflujo, y la última es los tokens en los que se divide el paquete
        lista_semiflujos = dstack(lista_semiflujos)
        lista_semiflujos = np.rollaxis(lista_semiflujos,-1) ## Necesario pues las dimensiones no estaban en el orden correcto

        print("Dimensionalidad de la lista de semiflujos:")
        print(lista_semiflujos.shape)

        if (normalizar == True): 
            lista_semiflujos = lista_semiflujos.astype('float32')
            for cont,semiflujo in enumerate(lista_semiflujos):
                lista_semiflujos[cont] = normalize(semiflujo,norm='max')
                
    ## Devolvemos los semiflujos encontrados (puede ser una lista vacía si no hemos encontrado nada)
    return lista_semiflujos
    


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------FUNCION CLASIFICAR_SEMIFLUJOS---------------------------------------------------------
###   Función que dado unos semiflujos y un modelo, realiza una clasificación de los mismos, imprimiendo un piechart con el
###   resultado
###   ------------------------------------------------------------------------------------------------------------------------

def clasificar_semiflujos(model,lista_semiflujos,batch_size,clases_clasificador,dict_total_semiflujos_clasificados):
    
    
    ## Si la lista contiene nuevos semiflujos, los clasificamos y añadimos al diccionario del total de semiflujos clasificados
    if (len(lista_semiflujos)!=0):
        ## Realizamos la prediccion del modelo sobre los semiflujos de esta categoria
        prediccion_categoria = model.predict(lista_semiflujos,batch_size=batch_size,verbose=0)

        ## Elegimos la clase ganadora, ya que el modelo devuelve probabilidad, asi que nos quedamos con la
        ## clase que tiene más probabilidad como ganadora
        prediccion_categoria_transformado = np.argmax(prediccion_categoria,axis=1)

        values, counts = np.unique(prediccion_categoria_transformado, return_counts=True)

        ## Convertimos a diccionario la predicción
        dict_prediccion = {values[i]: counts[i] for i in range(len(values))}
        dict_prediccion

        ## Añadimos la nueva predicción al total  
        dict_total_semiflujos_clasificados = dict(Counter(dict_prediccion)+Counter(dict_total_semiflujos_clasificados))
        dict_total_semiflujos_clasificados = dict(sorted(dict_total_semiflujos_clasificados.items()))
    
    ## Obtenemos los valores de todos los semiflujos clasificados hasta el momento, para realizar un piechart
    values = list(dict_total_semiflujos_clasificados.keys())
    counts = list(dict_total_semiflujos_clasificados.values())

    ## En values tenemos las categorias que han sido predichas, y en counts cuántos semiflujos por categoria
    ## Ej: values = [0,2,3] y counts = [10,15,5]
    ## El modelo habría identificado 10 semiflujos como categoria 0, 15 como categoria 2 y 5 como categoria 3

    ## Vamos a poner nombre a las categorias de values, es decir, cambiar el número por la clase de tráfico a
    ## a la que representan
    
    ## Borrar la pantalla entre piecharts
    clear_output(wait=True)

    labels_categorias = []

    for valor in values:
        labels_categorias.append(clases_clasificador[valor])

    lista_porcentajes = list()
    for item in counts:
        total = sum(counts)
        val = round(item/total*100,2)
        lista_porcentajes.append(val)

    plt.pie(counts)

    labels = ['%s - %s%% - %s semiflujos' % (l, s, c) for l, s, c in zip(labels_categorias, lista_porcentajes, counts)]
    plt.legend(bbox_to_anchor=(0.85, 1), loc='upper left', labels=labels)

    print("\nResultados obtenidos de la clasificación:")
    plt.show() 
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

    ## Devolvemos el diccionario actualizado con todos los semiflujos obtenidos hasta el momento
    return dict_total_semiflujos_clasificados


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 