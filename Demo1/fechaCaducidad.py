import boto3  # Importar el módulo boto3 para interactuar con los servicios de AWS
import json

# Definir el nombre de la función lambda
lambdaFunction = "LeerCaducidad"

# Definir el payload de la función lambda
payload = json.dumps({
    "s3BucketName": "onceaplicacion",
    "imageDocument": "yogurCaducidad1.jpg"
})
payload = payload.encode('utf-8')

# Función para crear una instancia del cliente de Textract
def crear_instancia_textract():
    # Ingresar las credenciales de AWS
    access_key = ""
    secret_key = ""
    region_name = ""
    
    # Crear una instancia del cliente de Textract con las credenciales y región especificadas
    textract_client = boto3.client('lambda', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region_name)
    return textract_client

# Función para consultar Textract y obtener las horas y fechas de caducidad
def consultar_textract(lambdaFunction, payload):
    textract_client = crear_instancia_textract()  # Crear instancia del cliente de Textract
    response = textract_client.invoke(
        FunctionName=lambdaFunction,
        Payload=payload
    )
    
    return  response['Payload'].read()  # Retornar las horas y fechas de caducidad extraídas


# Llamar a la función consultar_textract para obtener las horas y fechas de caducidad
response = consultar_textract(lambdaFunction, payload)

# Imprimir las horas de caducidad
print(response)

input("Presiona Enter para continuar...")