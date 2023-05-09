import sys
from pathlib import Path
import os

def main():
    try:
        # Código principal de tu programa
        # ...
        # Aquí es donde ocurre el error potencial
        import torch
        import numpy as np
        import cv2
        from time import time
        from ultralytics import YOLO
        from supervision import Detections, BoxAnnotator

        import pygame


        class ObjectDetection:

            def __init__(self, capture_index):
                # Inicializa la clase ObjectDetection con el índice de captura proporcionado
                self.capture_index = capture_index

                # Verifica si se puede utilizar la GPU para aceleración (utiliza 'cuda' si está disponible, de lo contrario, 'cpu')
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Carga el modelo YOLO pre-entrenado
                self.model = self.load_model()

                # Obtiene el diccionario de nombres de clases del modelo
                self.CLASS_NAMES_DICT = self.model.model.names

                # Crea un objeto BoxAnnotator para anotar las detecciones en los cuadros
                self.box_annotator = BoxAnnotator(thickness=3, text_thickness=3, text_scale=1.5)

            

            def load_model(self):
                # Carga el modelo YOLOv8n pre-entrenado desde el archivo "yolov8n.pt"
                model = YOLO("yolov8n.pt")

                # Fusiona las capas convolucionales del modelo para una mejor eficiencia de inferencia
                model.fuse()

                # Devuelve el modelo cargado
                return model



            def predict(self):
                """
                ALL CLASSES 
                39: 'bottle', 
                40: 'wine glass', 
                41: 'cup', 
                42: 'fork', 
                43: 'knife', 
                44: 'spoon', 
                45: 'bowl', 
                46: 'banana', 
                47: 'apple', 
                48: 'sandwich', 
                49: 'orange', 
                50: 'broccoli', 
                51: 'carrot', 
                52: 'hot dog', 
                53: 'pizza', 
                54: 'donut', 
                55: 'cake'
                """
                # Realiza la detección de objetos en el frame actual utilizando el modelo YOLO
                self.results = self.model(self.frame, classes=[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55])

                # Obtiene los resultados de la primera imagen en los resultados
                self.results = self.results[0]


            def get_processed_data(self):
                # Crear el diccionario de resultados
                object_locations = []

                # Extraer detecciones para todas las clases importantes
                for result in self.results:
                    boxes = []
                    for box in result.boxes.xyxy.cpu().numpy():
                        # Calcular la ubicación en la grid
                        h_position = int(((box[0] + box[2]) / 2) // self.h_limit)
                        v_position = int(((box[1] + box[3]) / 2) // self.v_limit)
                        
                        # Asignar la ubicación al diccionario de resultados
                        boxes.append((v_position, h_position))
                    object_locations.append(boxes)

                # Configurar detecciones para la visualización
                detections = Detections(
                    xyxy=self.results.boxes.xyxy.cpu().numpy(),
                    confidence=self.results.boxes.conf.cpu().numpy(),
                    class_id=self.results.boxes.cls.cpu().numpy().astype(int),
                )

                # Formatear etiquetas personalizadas
                self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                            for _, _, confidence, class_id, _ in detections]

                # Anotar y mostrar el frame
                self.frame = self.box_annotator.annotate(scene=self.frame, detections=detections, labels=self.labels)

                self.object_locations = object_locations

            def play_mp3(self):
                loc = self.object_locations[0][0]  # Obtiene la ubicación del primer objeto detectado
                self.mixer.music.load(self.voices[loc[0]][loc[1]])  # Carga el archivo MP3 en el reproductor de música
                self.mixer.music.play()  # Reproduce el archivo MP3

            def __call__(self):

                self.voices = [[ "Demo2\DebajoDerecha.mp3","Demo2\Debajo.mp3",  "Demo2\DebajoIzquierda.mp3"],
                                ["Demo2\Derecha.mp3",      "Demo2\Centrado.mp3","Demo2\Izquierda.mp3"],
                                ["Demo2\ArribaDerecha.mp3","Demo2\Arriba.mp3",  "Demo2\ArribaIzquierda.mp3"]]

                self.mixer = pygame.mixer
                self.mixer.init()

                cap = cv2.VideoCapture(self.capture_index)  # Inicializa la captura de video
                assert cap.isOpened()  # Asegura que la captura esté abierta correctamente
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Establece el ancho del marco de video
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Establece el alto del marco de video
                _, self.frame = cap.read()  # Lee el primer marco de video
                frame_height, frame_width, _ = self.frame.shape  # Obtiene el tamaño del marco de video
                self.h_limit = frame_width // 3  # Calcula el límite horizontal para la cuadrícula
                self.v_limit = frame_height // 3  # Calcula el límite vertical para la cuadrícula

                self.class_names = {39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
                        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                        54: 'donut', 55: 'cake'}

                sec_per_play = 5  # Duración en segundos entre reproducciones de audio
                play = 0
                while True:
                    
                    start_time = time()  # Registra el tiempo de inicio del ciclo

                    ret, self.frame = cap.read()  # Lee un nuevo marco de video
                    assert ret  # Asegura que se haya leído correctamente

                    # Reflejar la imagen horizontalmente
                    self.frame = cv2.flip(self.frame, 1)
                    
                    self.predict()  # Realiza la predicción de objetos en el marco de video
                    self.get_processed_data()  # Procesa los datos de detección obtenidos
                    
                    end_time = time()  # Registra el tiempo de finalización del ciclo
                    fps = 1/np.round(end_time - start_time, 2)  # Calcula los FPS (cuadros por segundo) del ciclo actual
                        
                    cv2.putText(self.frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)  # Agrega el texto FPS al marco de video
                    
                    cv2.imshow('YOLOv8 Detection', self.frame)  # Muestra el marco de video con las detecciones realizadas

                    if cv2.waitKey(5) & 0xFF == 27:  # Espera a que se presione la tecla 'Esc' para salir del bucle
                        break

                    if play >= fps*sec_per_play and self.object_locations:  # Verifica si ha pasado el tiempo suficiente y existen ubicaciones de objetos
                        self.play_mp3()  # Reproduce el archivo de audio correspondiente a la ubicación del objeto
                        play = -1  # Reinicia el contador de reproducción

                    play += 1  # Incrementa el contador de reproducción en cada iteración
                
                cap.release()  # Libera los recursos de captura de video
                cv2.destroyAllWindows()  # Cierra todas las ventanas de visualización
                
                
            
        detector = ObjectDetection(capture_index=0)
        detector()
    except Exception as e:
        # Captura cualquier excepción que ocurra y muestra un mensaje de error
        print("Se produjo un error:", e)
        # Opcionalmente, puedes pausar la ejecución para que el usuario pueda leer el mensaje
        input("Presiona Enter para continuar...")

if __name__ == '__main__':
    main()



