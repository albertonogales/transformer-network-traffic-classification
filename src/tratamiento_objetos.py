#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "tratamiento_objetos.py".

AUTOR: Carlos Arranz Luque.

FECHA: primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'tratamiento objetos', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "GUARDAR_DICCIONARIO_DFS"
    -FUNCION "GUARDAR_PESOS_MODELO"
    -FUNCION "OBTENER_PESOS_MODELO"    
    -FUNCION "CARGAR_DICCIONARIO_DFS"
    -FUNCION "CARGAR_SEMIFLUJOS"
    -FUNCION "GUARDAR_SEMIFLUJOS"

    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import os
import pickle
from src import md5hash

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION GUARDAR_DICCIONARIO_DFS-------------------------------------------------
###   Función que guarda en el equipo en la carpeta que recibe como argumento el conjunto de dataframes preprocesados que 
###   contiene el diccionario que recibe como argumento. Estos se guardan con el nombre 'DFS_categoria.obj'
###   ------------------------------------------------------------------------------------------------------------------------

def guardar_diccionario_DFS(DF_diccionario,pathRelativoDFS):
    
    for key in DF_diccionario:
        file_dict_semiflujos = open(pathRelativoDFS + 'DFS_'+key+'.obj', 'wb')
        ## Si estoy en la categoría "Chat", guardaré el archivo "DFS_Chat.obj" en el directorio "pathRelativoDFS"
        pickle.dump(DF_diccionario[key], file_dict_semiflujos)
        file_dict_semiflujos.close()

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION GUARDAR_PESOS_MODELO-------------------------------------------------
###   Función que guarda en el directorio indicado los pesos que recibe de un modelo de IA, se guardan también datos extras.
###   Listado de contenido del dic_modelo:
###      - Pesos del modelo
###      - Tiempo de entrenamiento (de un solo fold)
###      - Información del equipo: CPU, RAM, picoRAM, GPU, VRAM, picoVRAM
###      - Métricas del entrenamiento: media_train_accuraccy, media_train_loss, media_val_accuraccy, media_val_loss,
###                                    desviación_train_accuracy, desviación_train_loss, desviación_val_accuraccy,
###                                    desviación_val_loss
###   En total son 16 claves las que almacena el diccionario.
###   ------------------------------------------------------------------------------------------------------------------------

def guardar_pesos_modelo(hashSemiflujo,hashModelo, dic_modelo, pathRelativo, tipoModelo):
    
    pathCompleto = pathRelativo + 'HS_'+ hashSemiflujo +'_HM_'+hashModelo+'_Pesos'+tipoModelo+'.obj'

    ## Ejemplo con hashSemiflujo=12345, hashModelo=9876, y tipoModelo=XFMR:
    ## pathCompleto = "Objetos preprocesados/Pesos_TransformerSimple/HS_12345_HM_9876_PesosXFMR.obj"
    
    ## Ejemplo con hashSemiflujo=12345, hashModelo=9876, y tipoModelo=LSTM:
    ## pathCompleto = "Objetos preprocesados/Pesos_LSTM/HS_12345_HM_9876_PesosLSTM.obj"
    
    file_pesos_modelo = open(pathCompleto, 'wb')
    pickle.dump(dic_modelo, file_pesos_modelo)
    file_pesos_modelo.close()


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------




###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION OBTENER_PESOS_MODELO-------------------------------------------------
###   Función que obtiene los pesos y el resto de información de un modelo entrenado.
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_pesos_modelo(hashSemiflujo, hashModelo, pathRelativo, tipoModelo):
       
    pathCompleto = pathRelativo + 'HS_'+ hashSemiflujo +'_HM_'+hashModelo+'_Pesos'+tipoModelo+'.obj'
    
    ## Ejemplo con hashSemiflujo=12345, hashModelo=9876, y tipoModelo=XFMR:
    ## pathCompleto = "Objetos preprocesados/Pesos_TransformerSimple/HS_12345_HM_9876_PesosXFMR.obj"
    
    ## Ejemplo con hashSemiflujo=12345, hashModelo=9876, y tipoModelo=LSTM:
    ## pathCompleto = "Objetos preprocesados/Pesos_LSTM/HS_12345_HM_9876_PesosLSTM.obj"

    file_pesos_modelo = open(pathCompleto, 'rb')
    dic_modelo = pickle.load(file_pesos_modelo)
    file_pesos_modelo.close()
    
    return dic_modelo


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION CARGAR_DICCIONARIO_DFS-------------------------------------------------
###   Función que inicializa un diccionario y lo carga con los dataframes preprocesados que están guardados en el path que 
###   que recibe como argumento. Recibe también como argumento que categorías debe cargar de las posibles.
###   ------------------------------------------------------------------------------------------------------------------------

def cargar_diccionario_DFS(categorias_a_procesar, pathRelativoDFS):
    
    DF_diccionario = {}

    for categoria in categorias_a_procesar:
        try:
            file_df_procesado = open(pathRelativoDFS+'DFS_'+categoria+'.obj', 'rb')
        except:
            print("----------------------------------------------------------------------------------------")
            print("No existe el siguiente dataframe preprocesado: DFS_"+categoria+".obj")
            print("----------------------------------------------------------------------------------------")
            sys.exit()
        #Añadimos el listado de DFS al diccionario
        DF_diccionario[categoria] = pickle.load(file_df_procesado)
        file_df_procesado.close()


    return DF_diccionario

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION CARGAR_SEMIFLUJOS------------------------------------------------------
###   Función que recibe un path donde buscar y cargar los semiflujos. Dichos semiflujos se identifican con el hash que 
###   recibe como argumento.
###   Se cargan 3 archivos: el array tridimensional que contiene los semiflujos, el array de etiquetas de esos semiflujos, y
###   el diccionario que informa del contenido del array de semiflujos.
###   ------------------------------------------------------------------------------------------------------------------------

def cargar_semiflujos(hashSemiflujo, pathRelativo):
    
    pathSemicompleto = pathRelativo+hashSemiflujo+'/'

    file_array_semiflujos = open(pathSemicompleto+'array_tridimensional.obj', 'rb')
    array_semiflujos = pickle.load(file_array_semiflujos)
    file_array_semiflujos.close()
    
    file_array_etiquetas = open(pathSemicompleto+'array_etiquetas.obj', 'rb')
    array_etiquetas = pickle.load(file_array_etiquetas)
    file_array_etiquetas.close()
    
    file_diccionario_semiflujos = open(pathSemicompleto+'diccionario_semiflujos.obj', 'rb')
    diccionario_semiflujos = pickle.load(file_diccionario_semiflujos)
    file_diccionario_semiflujos.close()


    return array_semiflujos,array_etiquetas,diccionario_semiflujos

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   -------------------------------------------FUNCION GUARDAR_SEMIFLUJOS--------------------------------------------------
###   Función que guarda en el path que recibe como argumento la lista de semiflujos, sus etiquetas, su diccionario de
###   contenido y el archivo de configuración utilizado. Todos estos archivos son nombrados con un prefijo, un hash. 
###   Así podemos generar distintos ficheros de semiflujos en distintas pruebas y tenerlo organizado.
###   ------------------------------------------------------------------------------------------------------------------------

def guardar_semiflujos(Semiflujos_diccionario_contenido, lista_semiflujos, lista_arrays_etiqueta, pathRelativoSemiflujos, hashSemiflujos):

 
    pathSemiflujo = pathRelativoSemiflujos+'/'+hashSemiflujos
    ## Comprobamos si existe la carpeta que debe contener este tipo de semiflujos
    isExist = os.path.exists(pathSemiflujo)
    if not isExist:
        # La creamos si no existía
        os.makedirs(pathSemiflujo)
        print("Directorio para los semiflujos "+hashSemiflujos+" creado!")
    else:
        print("Directorio para los semiflujos "+hashSemiflujos+" ya existente! Contenido actualizado")
    
    ## Guardamos el diccionario de contenido usando el nombre correspondiente y dando .obj de extensión
    file_semiflujos = open(pathRelativoSemiflujos+'/'+hashSemiflujos + '/'+'diccionario_semiflujos.obj', 'wb') 
    pickle.dump(Semiflujos_diccionario_contenido, file_semiflujos)
    file_semiflujos.close()
    
    ## Guardamos la lista de semiflujos usando el nombre correspondiente y dando .obj de extensión
    file_semiflujos = open(pathRelativoSemiflujos+'/'+hashSemiflujos + '/'+'array_tridimensional.obj', 'wb') 
    pickle.dump(lista_semiflujos, file_semiflujos)
    file_semiflujos.close()
    
    ## Guardamos el array de etiquetas usando el nombre correspondiente y dando .obj de extensión
    file_semiflujos = open(pathRelativoSemiflujos+'/'+hashSemiflujos + '/'+'array_etiquetas.obj', 'wb') 
    pickle.dump(lista_arrays_etiqueta, file_semiflujos)
    file_semiflujos.close()
    
    
    ## Ahora vamos a copiar el archivo de configuración, renombrándolo con el hash obtenido
    comando = "cp config/config_preprocesamiento.cfg '"+pathRelativoSemiflujos+'/'+hashSemiflujos+"/config.cfg'"

    stream = os.popen(comando)

    ## Ejecutamos el comando (>.<)
    stream.read()
    

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 