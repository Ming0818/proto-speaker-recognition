# Speaker Recognition Prototype

Para el modelo de [speaker-recognition](https://github.com/ppwwyyxx/speaker-recognition) se uso el repositorio de https://github.com/ppwwyyxx/speaker-recognition el cual es realizado por medio de Gaussian Mixture Models.

## Guia de installación de speaker-recognition.
Para simplificar el proceso de instalación del speaker-recognitions primero debemos instalar [docker](https://docs.docker.com/install/), una vez instalado docker procedemos a descargar la imagen docker del repositorio de **speaker-recognition**

    docker pull qacollective/ppwwyyxx-speaker-recognition

Esto nos ahorra muchos problemas de incompatibilidad.

### uso de speaker recognition.
El speaker-recognition debe tener acceso a los directorios de nuestro ordenador, para ello usamos.

    sudo docker run --name speaker-recognitionInstance -ti -v /:/host speaker-recognition

Una vez hecho esto, entramos en el contenedor docker para entrenar el modelo.

    sudo docker start -ai speaker-recognitionInstance

copiamos el script speaker-recognition.py dentro del contenedor.

   sudo docker cp speaker-recognition.py speaker-recognitionInstance:/root/speaker-recognition/src/speaker-recognition.py

Procedemos a entrenar el modelo.

    cd /root/speaker-recognition/src/
    ./speaker-recognition.py -t enroll -i "/host/path/to/samples/label1 /host/path/to/samples/label2 " -m model.out

label1 y label2 serán las etiquetas las personas que queremos identificar.

Para rechazar muestras se debe entrenar un UBM, usando el comando de la siguiente forma logramos tener un UBM dentro de nuestro modelo UBM.

    ./speaker-recognition.py -t enroll -i "/host/path/to/samples/label1 /host/path/to/samples/label2 " -u "/host/path/to/samples/ubm" -m model.out

para clasificar las muestras dentro del modelo.

    ./speaker-recognition.py -t enroll -i "/host/path/to/samples/label1 /host/path/to/samples/label2 " -m model.out

### prueba.sh
este script de bash esta sujeto a cambios por ello el nombre.
antes de ejecutar este script se debe cambiar la líneas.

  docker start   container_id

  docker exec -it container_id root/speaker-recognition/src/speaker-recognition.py -t predict -i "/host/home/ricardo/Documents/TFM/codigo/   predict/*.wav" -m root/speaker-recognition/src/model.out

para saber la containder id que se generó tras hacer el pull de la imagen se usa el comando.

    sudo docker ps -aqf "name=speaker-recognitionInstance"

- q para que solo muestre el container id
- a para todos. funciona inclusive si no esta trabajando el contenedor
- f para filtrar

mode de ejecutar el script.

    sudo bash prueba.sh /path/to/audio.wav

esto clasificara las muestras del audio indicado y si identifica algún sujeto capturara el fragmento de audio en el cual el sujeto identificado intervino y lo almacenara en la carpeta audio_to_txt.

pd. las muestras de entrenamiento y clasificación deben pasar por el mismo filtro.
el script **split_audio.py** separa las muestras de audio cuando detecta silencio. además remueve las muestras que no detecta actividad de voz ya que las parte de silencio afectan negativamente la predicción del modelo.

por lo que es importante que las muestras de audio que se van a entrenar pasen por este filtro antes de ser entrenadas y clasificadas por el speaker recognition.

## Autores

- [Ricardo Canar](http://github.com/ricardocanar).

## Licencia

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
