<p align="center">
<img src="https://i.imgur.com/mgc7iqA.png" alt="Logo" width="100" height="100">
<br>
<img src="https://elreferente.es/wp-content/uploads/2021/08/LogoBlack_large.png" alt="LogoFOQUM" width="400" height="85" >
<p/>

# <p align="center">PROYECTO DEL CEIEC - Network Traffic Classification, en colaboración con Foqum</p>

### <p align="center">Becario responsable del proyecto: Carlos Arranz Luque</p>
### <p align="center">Directora de la investigación: Ana María Maitín López</p>
### <p align="center">Fecha: curso académico 2022-2023</p>
   
## 

# Descripción del proyecto
El objetivo del proyecto es la investigación del análisis y clasificación de tráfico de redes informáticas. Una red informática está compuesta por un conjunto de dispositivos electrónicos, los cuales están conectados entre sí 
e intercambian información; los mensajes que contienen dicha información componen el tráfico de la red. Clasificar satisfactoriamente dicho flujo de datos será la principal meta a lograr, identificando la categoría de comunicación, p. ej., correo electrónico, transferencia de archivos, geolocalización, etc.

El proyecto se nutre de datos recibidos por un sniffer de red como Wireshark, los cuales son programas en escucha permanente que capturan todo el tráfico de la red a la que están conectados. Wireshark en concreto, captura el tráfico y lo almacena en archivos "PCAP", dichos archivos formarán nuestro dataset de partida.

El proyecto ha ideado un flujo de ejecución que recibe un archivo o conjuntos de archivos "PCAP", de los
cuales extrae las características más relevantes, obteniendo "semiflujos", que son conjuntos de paquetes.
Dichos "semiflujos" serán la entrada de nuestro modelos neuronales. 

Existe un Jupyter Notebook para el preprocesamiento de los archivos "PCAP" y uno para cada modelo 
neuronal implementado (actualmente un LSTM y un Transformer).

<img src="https://github.com/ufvceiec/Network-traffic-classification-CEIEC/blob/develop_carlos/Imagenes/Esquema%20flujo%20de%20trabajo.png" width="1000" height="728">

# Uso y despliegue
Dentro de la carpeta "Docker_NTC" existe todo lo necesario para desplegar el proyecto fácilmente en Docker.
Próximamente se adjuntará un vídeo de YouTube explicando paso a paso las distintas partes del proyecto.

# Directorios y contenido
<img src="https://github.com/ufvceiec/Network-traffic-classification-CEIEC/blob/develop_carlos/Imagenes/Plantilla%20carpetas%20proyecto.png" width="520" height="953">
