#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
"""
NOMBRE DEL ARCHIVO: "metricas_modelo.py".

AUTOR: Carlos Arranz Luque.

FECHA: primer cuatrimestre de 2023.

DESCRIPCIÓN: módulo de Python que contiene funciones agrupadas en la categoría de 'obtención de métricas', para el proyecto
             de clasificación de tráfico de red.

LISTA DE LAS FUNCIONES QUE CONTIENE EL ARCHIVO:
    -FUNCION "CREAR_DIRECTORIO_GRAFICAS"
    -FUNCION "OBTENER_METRICAS_MODELO"
    -FUNCION "GUARDAR_RESULTADOS_MODELO"
    
"""
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######
#####-----------------------------------------------------------------------------------------------######

#####---------------------LIBRERIAS-----------------------------------------------------------------###### 

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#####---------------------FIN LIBRERIAS-------------------------------------------------------------###### 




#####---------------------FUNCIONES-----------------------------------------------------------------###### 


###   ------------------------------------------------------------------------------------------------------------------------
###   ---------------------------------------FUNCION CREAR_DIRECTORIO_GRAFICAS------------------------------------------------
###   Función que comprueba si ya existe un directorio para un modelo más semiflujos en concreto, si no existe lo crea
###   Recibe como argumentos el hash de los semiflujos y del modelo, así como la ruta donde debe comprobar si existe 
###   la carpeta
###   ------------------------------------------------------------------------------------------------------------------------

def crear_directorio_graficas(hashSemiflujos,hashModelo,rutaResultados):

    pathDirectorio = rutaResultados+'Graficas de entrenamiento/'+'HS_'+hashSemiflujos+'_HM_'+hashModelo
    # Comprobamos si existe o no el directorio
    isExist = os.path.exists(pathDirectorio)
    if not isExist:
        # No existe y lo creamos
        os.makedirs(pathDirectorio)
        print("Directorio creado!")
    else:
        print("Directorio ya existente!")

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------





###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION OBTENER_METRICAS_MODELO------------------------------------------------
###   Función que recibe el conjunto de predicciones realizadas por un modelo y la lista real que debería haber predicho.
###   En base a esto obtiene un conjunto de métricas para problemas de clasificación.
###   También recibe como argumento las métricas obtenidas durante el entrenamiento del modelo: accuracy y loss sobre training
###   y sobre validación, así como su desviación estándar.
###   Si la opción de guardar resultados está a "True" guarda la matriz de confusión en la "rutaResultados" recibida, dentro
###   del subdirectorio de matrices de confusión.
###   ------------------------------------------------------------------------------------------------------------------------

def obtener_metricas_modelo(y_test,y_pred,accuracy_test,loss_test,accuracy_train,loss_train,accuracy_val,loss_val,desviacion_train_acurracy,desviacion_train_loss,desviacion_val_acurracy,desviacion_val_loss,hashSemiflujos,hashModelo, rutaResultados, guardarResultados,categorias_semiflujos):
    
    
    ## Diccionario donde se almacenará el conjunto de métricas
    dic_metricas = {}
    
    y_test_transformado = np.argmax(y_test,axis=1)
    y_pred_transformado = np.argmax(y_pred,axis=1)   
    ## Quitamos las probabilidades, nos quedamos solo con la clase ganadora, una lista de qué clase debería haber salido
    ## y otra con la lista predicha por el modelo. (Este cambio es necesario para algunas métricas, que no admiten one
    ## hot encoding.)
    ## Ejemplos de la transformación:
    ##
    ## y_pred[0] = [0.2, 0.7, 0.1]       
    ## y_pred_transformado[0] = 1
    ##
    ## y_test[0] = [1, 0, 0]       
    ## y_test_transformado[0] = 0
    
    print("Las métricas obtenidas para el modelo son las siguientes:")
    
    print("\nAccuracy_test = " + str(round(accuracy_test,4)))
    dic_metricas["accuracy_test"] = round(accuracy_test,4)
    
    print("\nLoss_test = " + str(round(loss_test,4)))
    dic_metricas["loss_test"] = round(loss_test,4)
    
    print("\nAccuracy_training = " + str(round(accuracy_train,4)))
    dic_metricas["accuracy_training"] = round(accuracy_train,4)
    
    print("\nLoss_training = " + str(round(loss_train,4)))
    dic_metricas["loss_training"] = round(loss_train,4)
    
    print("\nAccuracy_validation = " + str(round(accuracy_val,4)))
    dic_metricas["accuracy_val"] = round(accuracy_val,4)
    
    print("\nLoss_validation = " + str(round(loss_val,4)))
    dic_metricas["loss_val"] = round(loss_val,4)
       
    #####
    
    print("\nDesviacion_train_acurracy = " + str(round(desviacion_train_acurracy,4)))
    dic_metricas["desviacion_train_acurracy"] = round(desviacion_train_acurracy,4)
    
    print("\nDesviacion_train_loss = " + str(round(desviacion_train_loss,4)))
    dic_metricas["desviacion_train_loss"] = round(desviacion_train_loss,4)
    
    print("\nDesviacion_val_acurracy = " + str(round(desviacion_val_acurracy,4)))
    dic_metricas["desviacion_val_acurracy"] = round(desviacion_val_acurracy,4)
    
    print("\nDesviacion_val_loss = " + str(round(desviacion_val_loss,4)))
    dic_metricas["desviacion_val_loss"] = round(desviacion_val_loss,4)
    
    ###### 
    
    cohen = cohen_kappa_score(y_test_transformado, y_pred_transformado) 
    print("\nCohen kappa score = " + str(round(cohen,4)))
    dic_metricas["cohen"] = round(cohen,4)
    
    matthews = matthews_corrcoef(y_test_transformado, y_pred_transformado)
    print("\nMatthews correlation coefficient = " + str(round(matthews,4)))
    dic_metricas["matthews"] = round(matthews,4)
    
    log = log_loss(y_test, y_pred)
    print("\nLog loss = " + str(round(log,4)))
    dic_metricas["log"] = round(log,4)
    
    try:
        roc = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovr")
    except:
        print("\nERROR en el ROC AUC SCORE por tener clase/s sin ningun dato")
        roc = 0
    
    print("\nRoc AUC score = " + str(round(roc,4)))
    dic_metricas["roc"] = round(roc,4)
    
    print("\n\nGeneric classification report:\n")
    print(classification_report(y_test_transformado, y_pred_transformado, target_names = categorias_semiflujos))
    
    num_categorias = len(categorias_semiflujos)
     
    confusion = confusion_matrix(y_test_transformado, y_pred_transformado, labels = list(range(0, num_categorias)))
    print('\n\nConfusion Matrix\n')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=categorias_semiflujos)
    disp.plot()
    
    if(guardarResultados == True):
        ## Guardamos la matriz de confusión si la opción está seleccionada
        plt.savefig(rutaResultados+'Matrices de confusion/HS_'+hashSemiflujos+'_HM_'+hashModelo+'_ConfusionMatrix.png',facecolor='white',bbox_inches = 'tight',dpi=150)
        
    plt.show()
      
    return dic_metricas

###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------


###   ------------------------------------------------------------------------------------------------------------------------
###   -----------------------------------------FUNCION GUARDAR_RESULTADOS_MODELO----------------------------------------------
###   Función que recibe el diccionario con las métricas y las añade al archivo csv de métricas
###   
###   ------------------------------------------------------------------------------------------------------------------------

def guardar_resultados_modelo(dic_metricas, rutaResultados, hashModelo,cargar_pesos,tipoModelo):
    
    ## Cargamos en formato dataframe el CSV
    df = pd.read_csv(rutaResultados+"grid_"+tipoModelo+".csv")
    
    ## Añadimos en una nueva fila del dataframe nuestras métricas del entrenamiento actual
    df.loc[len(df)]=dic_metricas
    
    ## Sobrescribimos el CSV con el nuevo actualizado
    df.to_csv(rutaResultados+"grid_"+tipoModelo+".csv",index=False)
    
    ## Solo se ejecuta si es un entrenamiento que se acaba de realizar
    if (cargar_pesos == False):
        ## Ahora vamos a copiar el archivo de configuración, renombrándolo con el hash obtenido
        comando = "cp config/config_entrenamiento_"+tipoModelo+".cfg '"+rutaResultados+"Archivos de configuracion/"+hashModelo+".cfg'"

        stream = os.popen(comando)

        ## Ejecutamos el comando (>.<)
        stream.read()
        ## Hemos copiado y renombrado el archivo de configuración que ha usado este entrenamiento,
        ## de esta forma se podrá consultar en el futuro si lo necesitamos


###   ------------------------------------------------------------------------------------------------------------------------
###   --------------------------------------------------FIN FUNCION-----------------------------------------------------------
###   ------------------------------------------------------------------------------------------------------------------------



#####---------------------FIN FUNCIONES-------------------------------------------------------------###### 
