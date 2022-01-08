Los servicios son activados por medio del servidor y es la configuracion definida en el 
Dashboard la que se aplica sobre el servidor Nano especifico.

Las siguientes variables de ambiente son necesarios antes de ejecutar el script de busqueda de rostros.


1. $USER_SERVER_ENDPOINT:  Define la ruta del URI  (base “endpoint/web service”).
   Esta URI es la ruta hacia el servidor y sus servicios 
           
2. $FACE_RECOGNITION_TOKEN:  Esta variable define la ruta local del servidor Nano 
   hacia el archivo que contiene el token necesario para autenticarse ante el servidor 
   de servicios
       
3. $INPUT_DB_DIRECTORY: esta variable define el direcorio local en el servidor Nano 
   donde se crearan y almacenaran las diferentes bases de datos de los rostros a busca.
       
4. $RESULTS_DIRECTORY: esta variable define la ruta donde se almacenan los resultados 
   de los procesos


export USER_SERVER_ENDPOINT="https://mit.kairosconnect.app/"
export BASE_DIRECTORY=$HOME/"faceRecognition"
export FACE_RECOGNITION_TOKEN=$BASE_DIRECTORY/.token
export INPUT_DB_DIRECTORY=$BASE_DIRECTORY/data_input
export RESULTS_DIRECTORY=$BASE_DIRECTORY/face_results


*** Put the images of the black or white list subjects in data/black_list and/or data/white_list prior script execution

*** Use script load_subject.py to manually add the Black or White list. 

Usage: ./load_subject.py newBlackList | newWhitelist | addToBlackList 


