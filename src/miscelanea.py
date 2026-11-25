#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "miscelanea.py".

AUTOR: Carlos Arranz Luque.

FECHA: Primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'miscelánea', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "OBTENER_ESPECIFICACIONES_PC" 
    -FUNCION "GET_TIMESTAMP" 
    -FUNCION "CALCULAR_TIEMPO" 
    -FUNCION "IS_TOOL"
    -FUNCION "MD5HASH"
    -FUNCION "IMPRIMIR_CONTENIDO_ETIQUETAS_SET"

    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import time
import os
import datetime
import hashlib
import tensorflow as tf
import numpy as np
import pandas as pd
from shutil import which


#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 



#####---------------------FUNCIONES-----------------------------------------------------------------###### 





###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------FUNCION OBTENER_ESPECIFICACIONES_PC--------------------------------------------------
###   Función que imprime por pantalla y te devuelve la siguiente información:CPU,RAM,picoRAM,nombre_GPU,picoVRAM
###   - Nombre CPU
###   - Cantidad de RAM total del sistema
###   - Pico de uso de RAM del programa actual
###   - Nombre GPU (si TF la detecta)
###   - Cantidad de VRAM de la GPU 
###   - Pico de uso de VRAM de TF
###   
###   (También hace un print del PID del programa actual)
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_especificaciones_pc():
    
    ## (Todos los comandos de terminal, los que se ejecutan con el módulo "os", son de LINUX)
 
    ## Comando para obtener el modelo de CPU
    comando = "grep -m 1 'model name' /proc/cpuinfo"
    stream = os.popen(comando)
    CPU = stream.read()  
    CPU = CPU.split(":")[1][1:-1]
    
    ## Comando para obtener la RAM total del sistema (en KB)
    comando = "grep MemTotal /proc/meminfo"
    stream = os.popen(comando)
    RAM = stream.read()
    RAM = [int(s) for s in RAM.split() if s.isdigit()][0]
    RAM = int(round( (RAM/(1024*1000)) ,0))  ## Cambiar de KB a GB (mezclando base 2 y base 10)
    
    ## Comandos para obtener el pico de RAM usada
    PID = str(os. getpid())
    print("\nPID del entrenamiento = "+PID)
    comando = "grep VmHWM /proc/"+PID+"/status"   ## Comando que te dice el pico de RAM de un proceso
    stream = os.popen(comando)
    picoRAM = stream.read()  
    picoRAM = [int(s) for s in picoRAM.split() if s.isdigit()][0]
    picoRAM = round( (picoRAM/(1024*1000)) ,1)  ## Cambiar de KB a GB (sí, mezclando base 2 y base 10)
    ## (La mezcla hace que nos acerquemos de la mejor forma a la cantidad física instalada, por el lío entre GB y GiB)
      
    ## Comprobamos si tenemos GPU asociada a Tensorflow
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:  
        ## Comando para obtener el modelo de GPU
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        nombre_GPU = details.get('device_name') 
        
        ## Comando para obtener la VRAM total de la GPU (en MiB)
        ## -----------------------------------¡¡¡¡WARNING!!!!----------------------------------------
        ## Esto funciona mientras haya una sola gráfica en el sistema o todas las que haya sean iguales
        ## Si esto no es así hay que cambiarlo, ya que estoy usando nvidia-smi y quedándome con la
        ## información de la primera gráfica que me sale en el sistema
        ## -----------------------------------¡¡¡¡WARNING!!!!----------------------------------------
        comando = "nvidia-smi | grep -m 1 MiB"
        stream = os.popen(comando)
        VRAM = stream.read()       
        listaVRAM = VRAM.split()
        ## Hago split de todas las palabras obtenidas del comando "nvidia-smi | grep -m 1 MiB" y me quedo 
        ## con la segunda que tenga la palabra MiB, que es la que indica la cantidad de VRAM de la gráfica
        segunda = False
        for palabra in listaVRAM:
            if ("MiB" in palabra):
                if(segunda == True):
                    VRAM = palabra
                segunda = True
                 
        VRAM = VRAM[:-3]  ## Me quedo solo con la parte numérica
        VRAM = int(float(VRAM)/1024)  ## Transformación a GiB
        
        ## Comando para obtener el pico de VRAM usada
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:  ## No pongo else, ya que si llegamos aqui es que hay GPU 
            ## -----------------------------------¡¡¡¡WARNING!!!!----------------------------------------
            ## Habrá que confirmar para equipos con más de una GPU en el mismo docker, que número de GPU
            ## es la que estás usando. (Modificar la funcion y añadir un argumento con esa información)
            ## -----------------------------------¡¡¡¡WARNING!!!!----------------------------------------
            picoVRAM = tf.config.experimental.get_memory_info('GPU:0')['peak']
            picoVRAM = round((picoVRAM/(1024*1024*1024)),1)  ## Pasamos de bytes a GiB, con 1 decimal
      
    ## No hemos detectado GPU en el else
    else:
        nombre_GPU = "WARNING: GPU no detectada"
        VRAM = "N/A"
        picoVRAM = "N/A"
        
        
    print("\nCaracterísticas del PC donde se ha ejecutado el entrenamiento:")
    print("- CPU = "+CPU)
    print("- RAM = "+str(RAM)+"GB")
    print("- Pico de uso de RAM = "+str(picoRAM)+"GB")
    print("- GPU = "+nombre_GPU)
    print("- VRAM = "+str(VRAM)+"GB")
    print("- Pico de uso de VRAM = "+str(picoVRAM)+"GB")
    
    
    return CPU,RAM,picoRAM,nombre_GPU,VRAM,picoVRAM
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------FUNCION GET_TIMESTAMP-----------------------------------------------------------
###   Función que devuelve un string con el timestamp del momento de la ejecución, 
###   Formato --> "día-mes-año horas:minutos:segundos"
###   ------------------------------------------------------------------------------------------------------------------------

def get_timestamp():
    t = datetime.datetime.now()
    timestamp = t.strftime('%d-%m-%Y %H:%M:%S.%f') 
    return timestamp[:-7] ## Devolvemos todo menos los decimales de los segundos :3
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------FUNCION CALCULAR_TIEMPO-----------------------------------------------------------
###   Función que dado un tiempo inicial que recibe como argumento te imprime por pantalla cuánto tiempo ha pasado 
###   desde entonces. (Hace return de los minutos también por si es necesario guardar la información)
###   Si "info_visual" es "False", solo hace el return, sin imprimir nada por pantalla.
###   Si "clasificación_tiempo_real" es "True", imprime por pantalla los segundos transcurridos.
###   ------------------------------------------------------------------------------------------------------------------------

def calcular_tiempo(start_time, info_visual = True, clasificación_tiempo_real = False):
    
    if(clasificación_tiempo_real == True):
        
        tiempo = round(time.time() - start_time, 2)
        print("\nLa clasificación ha sufrido un delay de "+str(tiempo)+" segundos")
    
    else:
        minutos_ejecucion = int( (time.time() - start_time)/60)

        if(info_visual == True):
            segundos_ejecucion = int ((time.time() - start_time)%60)
            print("\n\n--------------------------------------------------------------------------------------------------")
            print("Este módulo de código ha tardado "+str(minutos_ejecucion)+" minutos y "+str(segundos_ejecucion)+" segundos en ejecutarse")
            print("--------------------------------------------------------------------------------------------------")

        return minutos_ejecucion
    
###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



###   ---------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------------FUNCION IS_TOOL--------------------------------------------------------
###   Función que comprueba si un programa está instalado en el sistema.
###   Devuelve "True" o "False".
###   Obtenida de StackOverflow:
###   https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script   
###   ---------------------------------------------------------------------------------------------------------------------

def is_tool(name):
    """Check whether 'name' is on PATH and marked as executable."""
    return which(name) is not None

### ------------------------------------------------------------------------------------------------------------------------
### --------------------------------------------------FIN FUNCION IS_TOOL---------------------------------------------------
### ------------------------------------------------------------------------------------------------------------------------



###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------------------FUNCION MD5HASH-----------------------------------------------------------
###   Función que genera un hash de tipo md5 dado un string que recibe como argumento
###   ------------------------------------------------------------------------------------------------------------------------

def md5hash(s: str):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   ----------------------------------FUNCION IMPRIMIR_CONTENIDO_ETIQUETAS_SET----------------------------------------------
###   Función que recibe un set de etiquetas, en formato one hot encoding, y muestra el contenido y porcentajes de cada clase
###   del mismo.
###   ------------------------------------------------------------------------------------------------------------------------

def imprimir_contenido_etiquetas_set(set_datos, nombre_set): 
    
    set_datos_transformado = np.argmax(set_datos,axis=1) ## Transformamos de one hot encoding a una única dimensionalidad
    ## Ejemplo:
    ## set_datos[0] = [1, 0, 0]       
    ## set_datos_transformado[0] = 0
        
    unique, counts = np.unique(set_datos_transformado, return_counts=True)
    contenido_set = dict(zip(unique, counts)) ## Diccionario con las ocurrencias de cada clase

    print("\n\nInformación del set "+nombre_set+":\n")
    for key in contenido_set:
        num_sem_total = set_datos.shape[0]
        num_sem_cat = contenido_set[key]
        porcentaje = round((num_sem_cat/num_sem_total*100), 2)
        print("Para la categoría "+str(key)+" hay "+str(num_sem_cat)+" semiflujos de "+str(num_sem_total)+", lo que supone un "+str(porcentaje)+"% del total.")
    print("En total el set contiene "+str(num_sem_total)+" semiflujos.")

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------

#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 