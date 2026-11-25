#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "procesamiento_pcap.py".

AUTOR: Carlos Arranz Luque.

FECHA: Primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'procesamiento pcap', para el proyecto
             de clasificación de tráfico de red. Los archivos '.pcap'/'.pcapng' son archivos que contienen una grabación de
             tráfico de red. Programas como 'Wireshark' o 'Tshark' generan este tipo de archivos.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "PROCESARDF"
    -FUNCION "GENERAR_DFS"
    -FUNCION "PROCESAR_CSVS_A_DFS"    
    -FUNCION "GENERAR_CSV_TSHARK"
    -FUNCION "GENERAR_CSVS"
    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import os
import numpy as np
import pandas as pd
from src import md5hash

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 





###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------------FUNCION PROCESARDF--------------------------------------------------------
###   Función que recibe un dataframe como argumento, el cual modifica hasta dejarlo de la forma deseada para el proyecto
###   ------------------------------------------------------------------------------------------------------------------------

def procesarDF(df):
    # Vamos a eliminar las filas que no contegan alguno de los siguientes protocolos, ya que no aportan información útil
    # (ESTA LISTA DEBE REVISARSE SEGÚN SE CAPTURE TRÁFICO QUE PUEDA TENER OTROS PROTOCOLOS)
    listaProtocolos = ["TCP","UDP","SSL","SSLv2","TLSv1","TLSv1.1","TLSv1.2","TLSv1.3","SSHv2","GTP <TCP>","GTP <UDP>","GTP <QUIC>","GTP <SSL>","GTP <SSLv2>","GTP <TLSv1>","GTP <TLSv1.1>","GTP <TLSv1.2>","GTP <TLSv1.3>"]
    

    # El caracter "~" sirve para negar la salida de la función "isin", de tal forme que transformamos 'True' a 'False'
    # y viceversa.
    # La opción "inplace=True", que se usará muchas veces, sirve para indicar que el cambio se haga en el propio 
    # dataframe, en vez de devolver uno con las modificaciones aplicadas, así nos evitamos el siguiente código:
    # df = df.drop(df[~df.Protocol.isin(listaProtocolos)].index)
    df.drop(df[~df.Protocol.isin(listaProtocolos)].index, inplace=True)
    
    # Vamos a eliminar las filas que tienen un TCPlen menor o igual a 2, no aportan nada, son confirmaciones de recibido
    df.drop(df[df.TCPlen <= 2].index, inplace=True)
     
    # Tenemos que resetear el índice ya que no se actualiza al borrar filas
    df.reset_index(drop=True, inplace = True)
    
    # Las columnas que nos indican el flujo al que pertenece un paquete son ['IPsrc', 'IPdst', 'Protocol','TCPsrcport',
    # 'TCPdstport','UDPsrcport', 'UDPdstport'], así que generamos una nueva columna con la etiqueta del flujo al que
    # pertenece el paquete, con el contenido del conjunto de dichas columnas

    df["Etiqueta"] = df['IPsrc'].astype(str) +"-"+ df['IPdst'].astype(str) +"-"+ df['Protocol'].astype(str) +"-"+ df['TCPsrcport'].astype(str) +"-"+ df['TCPdstport'].astype(str) +"-"+ df['UDPsrcport'].astype(str) +"-"+ df['UDPdstport'].astype(str)
    
    ## Aplicamos el hash md5 a nuestra columna "Etiqueta" para generar nuestra nueva columna, así tendremos una
    ## etiqueta más visual para cada flujo
    df["Hash"]=df["Etiqueta"].apply(md5hash)
    

    ## Cambio para que algunas paquetes extraños no tengan valor en la columna 'UDPpayload' cuando son TCP
    ## Igual con la de la columna 'TCPpayload' para los que son UDP
    
    ## "GTP <QUIC>" es una versión moderna, que está por encima del protocolo UDP, pero que usa este mismo.
    ## De todos los protocolos admitidos, solo "GTP <QUIC>" y "UDP", usan el protocolo UDP y tienen por tanto
    ## la información de sus paquetes en la columna "UDPpayload"
    
    ## Si no soy "UDP" o "GTP <QUIC>", vacío el contenido de la columna "UDPpayload"
    df.loc[ (df["Protocol"] != "UDP") & (df["Protocol"] != "GTP <QUIC>"), "UDPpayload"] = np.nan
    
    ## Si soy "UDP" o "GTP <QUIC>", vacío el contenido de la columna "TCPpayload"
    df.loc[ (df["Protocol"] == "UDP") | (df["Protocol"] == "GTP <QUIC>"), "TCPpayload"] = np.nan
    
    ## Eliminamos columnas que ya no aportan nada
    df.drop(columns=['No.','Protocol','IPsrc','IPdst','Length','TCPsrcport','TCPdstport','TCPlen','UDPsrcport','UDPdstport','Etiqueta'],inplace=True)
    
    ## Algunos paquetes no contienen ni TCPpayload ni UDPpayload, los borramos, no nos sirven de nada
    df = df[~((df['TCPpayload'].isnull()) & (df['UDPpayload'].isnull()))]
    
    #Tenemos que resetear el índice ya que no se actualiza al borrar filas
    df.reset_index(drop=True, inplace = True)
           
    print("Dataframe transformado!")

    
    return df
    

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------------FUNCION GENERAR_DFS-----------------------------------------------------------
###   Función que genera una lista de dataframes, uno por cada fichero CSV que exista en el directorio que recibe 
###   como argumento.
###   Los argumento "tiempoReal" y "nombreCSV", se han añadido en una versión posterior, empleándose para la clasificación
###   en tiempo real.
###   ------------------------------------------------------------------------------------------------------------------------

def generar_DFs(pathCSV, tiempoReal = False, nombreCSV = None):
    
    ## Las columnas que deben usarse en el dataframe
    colnames=['No.', 'Time', 'IPsrc', 'IPdst', 'Protocol', 'Length', 'TCPsrcport', 'TCPdstport', 'TCPlen', 'UDPsrcport', 'UDPdstport', 'TCPpayload', 'UDPpayload'] 
    
    
    ## Fragmento de código para clasificación en tiempo real
    if(tiempoReal == True):
        df = pd.read_csv(pathCSV+'/'+nombreCSV, dtype={'tcp.payload': 'object','udp.payload': 'object'})

        df = df.set_axis(colnames, axis=1)

        ## Llamamos a la función que deja procesado de la forma que queremos el DF que acabamos de leer del CSV
        df = procesarDF(df)    

        return df
    ## Fin fragmento para clasificación de tiempo real
    
    else:
        ## Generamos una lista vacía donde iremos agregando los dataframes
        lista_DF= []


        ## Comprobamos qué ficheros existen en el directorio CSV que nos han pasado como argumento
        ficherosCSV=os.listdir(pathCSV)

        num_ficheros = len(ficherosCSV)

        if(num_ficheros == 0):
            print("\n\n------------ ERROR DE LA FUNCION 'generar_DFs' ------------")
            print("\nDirectorio de CSVs vacio!!")
            return lista_DF

        ## Vamos creando un DF por cada fichero CSV
        for fichero in ficherosCSV:

            ## No procesamos el ".gitignore", es el único fichero que no es un CSV
            if(fichero != '.gitignore'):

                print("El fichero "+str(fichero)+" VA A SER se ha adjuntado a la lista de DF")

                ## Generamos el DF
                ## Se pone el 'dtype' a 'object' en la columna TCPpayload y UDPpayload para que no de un warning en algunos CSV

                df = pd.read_csv(pathCSV+'/'+fichero, dtype={'tcp.payload': 'object','udp.payload': 'object'})

                df = df.set_axis(colnames, axis=1)

                ## Llamamos a la función que deja procesado de la forma que queremos el DF que acabamos de leer del CSV
                df = procesarDF(df)    

                ## Añadimos el DF a la lista
                lista_DF.append(df)

                print("El fichero "+str(fichero)+" se ha adjuntado a la lista de DF\n")


        return lista_DF

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------------FUNCION PROCESAR_CSVS_A_DFS----------------------------------------------------
###   Función que genera y devuelve un diccionario con listas de dataframes, una por cada categoría que de la lista 
###   de pathsCSV que recibe.
###   ------------------------------------------------------------------------------------------------------------------------

def procesar_csvs_a_dfs(listaPathCSV,categorias_a_procesar):
    
    ## Creamos un diccionario donde guardaremos los DataFrames
    DF_diccionario = {}
    
    ## Vamos a recorrer el listado de subdirectorios donde están los CSV 
    for pathCSV in listaPathCSV:

        procesar = False
        
        ## Vamos a comprobar si estamos en un path de alguna de las categorías a procesar
        for categoria in categorias_a_procesar:
            if (categoria in pathCSV):
                procesar = True
                nombre_categoria = categoria
                
        ## Si esta categoría está en la lista de categorías a procesar del archivo de configuración, la procesamos
        if (procesar == True):
            ## Esta ejecución tiene mucho feedback visual para que sepas cómo va avanzando ya que tarda bastante
            print("\n--------------------------------------------------------")
            print("Comenzando a crear diccionario de "+nombre_categoria)
            print("--------------------------------------------------------\n")
            ## Asignamos al diccionario el listado de DFs que nos devuelve la función
            DF_diccionario[nombre_categoria]=generar_DFs(pathCSV)
            print("\n--------------------------------------------------------")
            print("Diccionario de "+nombre_categoria+" creado!")
            print("--------------------------------------------------------\n\n")
            
            
    return DF_diccionario
    
    
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   ------------------------------------------FUNCION GENERAR_CSV_TSHARK----------------------------------------------------
###   Función que genera un archivo CSV a partir de un archivo PCAP, obteniendo la información relevante del mismo para
###   el preprocesamiento de nuestra modelo. 
###      
###   En el comando de Tshark seleccionamos la siguiente información:
###       - Número de paquete
###       - Instante de tiempo en el que se ha capturado el paquete
###       - Ip origen
###       - Ip destino
###       - Protocolo
###       - Tamaño del paquete (número de bytes)
###       - Puerto origen en caso de protocolo TCP, vacío en otro caso
###       - Puerto destino en caso de protocolo TCP, vacío en otro caso
###       - Tamaño del payload en caso de protocolo TCP, vacío en otro caso
###       - Puerto origen en caso de protocolo UDP, vacío en otro caso
###       - Puerto destino en caso de protocolo UDP, vacío en otro caso
###       - Contenido en bruto del payload en caso de protocolo TCP, vacío en otro caso
###       - Contenido en bruto del payload en caso de protocolo UDP, vacío en otro caso
###      
###   Toda esta información se redirige a un fichero CSV de salida.
###                   
###   ------------------------------------------------------------------------------------------------------------------------

def generar_CSV_Tshark(nombrePCAP, pathPCAP, pathCSV):
    
    # Generamos el string del comando Tshark
    # Especificamos el nombre del fichero PCAP, los campos que queremos del mismo y el fichero de salida
    comando_Tshark = "tshark -r '"+pathPCAP+"/"+nombrePCAP+"' -E separator=, -E occurrence=l -E header=y -Tfields -e _ws.col.No. -e _ws.col.Time -e _ws.col.Source -e _ws.col.Destination -e _ws.col.Protocol -e _ws.col.Length -e tcp.srcport -e tcp.dstport -e tcp.len -e udp.srcport -e udp.dstport -e tcp.payload -e udp.payload > '"+pathCSV+"/"+nombrePCAP+".csv'"
    
    
    # Cargamos el comando 
    stream = os.popen(comando_Tshark)
    
    # Ejecutamos el comando (>.<)
    output = stream.read()


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------




###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------------FUNCION GENERAR_CSVS------------------------------------------------------
###   Función que recibe el path donde se encuentran los ficheros PCAP de cierta categoría. Los convierte en ficheros CSV
###   uno a uno, mediante un comando de TShark.
###   Los argumento "tiempoReal" y "nombreCSV", se han añadido en una versión posterior, empleándose para la clasificación
###   en tiempo real.
###   ------------------------------------------------------------------------------------------------------------------------

def generar_CSVs(pathPCAP, pathCSV, tiempoReal = False, nombrePCAP = None ):
    
    
    ## Estoy en tiempo real y sólo quiero procesar un PCAP en específico
    if (tiempoReal == True):
        generar_CSV_Tshark(nombrePCAP,pathPCAP,pathCSV)
    
    else:
        # Listamos los ficheros que existan en el directorio pathPCAP
        ficherosPCAP=os.listdir(pathPCAP)

        num_ficheros = len(ficherosPCAP)

        if(num_ficheros == 0):
            print("\n\n------------ ERROR DE LA FUNCION 'generar_CSVs' ------------")
            print("\nDirectorio de PCAPs vacío!\n")
            return 

        for fichero in ficherosPCAP:
            if (fichero != '.gitignore'):
                print("\nEl fichero "+str(fichero)+" VA A SER transformado a CSV")
                # Para cada fichero PCAP que hemos encontrado, generamos su CSV con el comando correspondiente de TShark
                generar_CSV_Tshark(fichero,pathPCAP,pathCSV)
                print("El fichero "+str(fichero)+" se ha transformado a CSV correctamente")

        print("\n\nRealizada la conversión a CSV de "+str(num_ficheros)+" ficheros PCAP encontrados en el directorio "+pathPCAP)
        
        
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------







#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 