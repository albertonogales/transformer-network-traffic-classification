## -----------------------------------------------------------------------------------------------------------------------------
## -----------------------DOCKERFILE PARA EL PROYECTO DE CLASIFICACIÓN DE TRÁFICO DE RED----------------------------------------
## -------------------------------FECHA DE ACTUALIZACIÓN : JULIO DE 2023--------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------

## Elegimos sobre qué imagen de docker vamos a trabajar, importante fijar una versión y NO usar "latest", para asegurar
## que en el futuro la imagen que cree este dockerfile sigue funcionando. (Latest va cambiando con el tiempo)
FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

## Realizamos un update para poder instalar "sudo" y "tshark"
RUN apt update

## Añado un repositorio para instalar una versión reciente de "tshark"
RUN add-apt-repository -y ppa:wireshark-dev/stable

## Instalamos "sudo" y "tshark", paquetes necesarios para el proyecto de clasificación de tráfico de red.
## Aquí cada uno verá qué paquetes necesita.
## ¡¡AVISO!! No confundir con las librerías de python existentes en el archivo requirements.txt
RUN apt install sudo

## Instalamos "tshark" de forma que permite darle permisos de grabación a usuarios que no sean root
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tshark
RUN yes yes | DEBIAN_FRONTEND=teletype dpkg-reconfigure wireshark-common

## Permisos de ejecución
RUN chmod +x /bin/dumpcap

## Copiamos nuestro txt con los requisitos
COPY requirements.txt ./

## Instalamos las librerías de Python necesarias
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

## Eliminamos el archivo que habíamos copiado
RUN rm requirements.txt

## Vamos a crear un usuario, para tener un home propio y no trabajar con el usuario root. 
## Hay que asignar a este usuario que creamos el mismo ID de usuario que tengas en la máquina que hace de host,
## con esto evitamos posibles problemas de privilegios con los archivos que generemos.
## Para consultar tu UID en la máquina host, ejecuta el comando "id"
## También asignamos una clave al usuario, ya que será necesaria si queremos ejecutar el comando "sudo".
## En este ejemplo '12345' es la password que se asignará al usuario 'ceiec', y el UID será el 1000
RUN useradd -m -u 1000 ceiec && echo "ceiec:12345" | chpasswd

## Añadimos el usuario que acabamos de crear al grupo de usuarios que pueden ejecutar "sudo"
RUN usermod -aG sudo ceiec

## Añadimos el usuario que acabamos de crear al grupo de usuarios de wireshark
RUN usermod -aG wireshark ceiec

## Establecemos el directorio de trabajo predefinido cuando se arranque el contenedor
WORKDIR /home/ceiec/app/

## Establecemos el usuario predefinido cuando se arranque el contenedor
USER ceiec
