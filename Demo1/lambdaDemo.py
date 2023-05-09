import json

import boto3  # Importar el módulo boto3 para interactuar con los servicios de AWS
import re  # Importar el módulo re para trabajar con expresiones regulares

# Función para verificar si una cadena tiene el formato de hora válido
def check_hora(hora):
    return re.findall("^(2[0-3]|[01]?[0-9]):([0-5]?[0-9])(:([0-5]?[0-9]))?$",hora)

# Función para verificar si una cadena tiene el formato de fecha válido
def check_fecha(fecha):
    return re.findall(r'\b(?:(?:\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))[-/])?(?:(?:\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))[-/])?(?:\d{2}|\d{4})\b',fecha) and len(fecha)>2

# Función para extraer las horas y fechas de caducidad de un diccionario de texto
def extract_caducidad(textos):
    hora = []  # Lista para almacenar las horas de caducidad
    fecha = []  # Lista para almacenar las fechas de caducidad
    
    # Recorrer los bloques de texto en el diccionario
    for item in textos["Blocks"]:
        if item["BlockType"] == "LINE":  # Verificar si el bloque es de tipo línea
            if check_hora(item["Text"]):  # Verificar si el texto es una hora válida
                hora.append(item["Text"])  # Agregar la hora a la lista
            elif check_fecha(item["Text"]):  # Verificar si el texto es una fecha válida
                fecha.append(item["Text"])  # Agregar la fecha a la lista
    
    # Retornar las listas de horas y fechas sin duplicados
    return list(set(hora)), list(set(fecha))

# Función para crear una instancia del cliente de Textract
def crear_instancia_textract():
    # Ingresar las credenciales de AWS
    
    # Crear una instancia del cliente de Textract
    textract_client = boto3.client('textract')
    return textract_client

# Función para consultar Textract y obtener las horas y fechas de caducidad
def consultar_textract(s3BucketName, imageDocument):
    textract_client = crear_instancia_textract()  # Crear instancia del cliente de Textract
    response = textract_client.detect_document_text(
        Document={
            'S3Object': {
                'Bucket': s3BucketName,
                'Name': imageDocument
            }
        })
    
    return extract_caducidad(response)  # Retornar las horas y fechas de caducidad extraídas

def lambda_handler(event, context):
    # TODO implement
    # Llamar a la función consultar_textract para obtener las horas y fechas de caducidad
    hora, fecha = consultar_textract(event["s3BucketName"], event["imageDocument"])
    
    return {
        'statusCode': 200,
        'body': json.dumps(str(hora[0])+" "+str(fecha[0]))
    }