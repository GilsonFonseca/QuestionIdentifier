import __init__
from client_python_m2p import M2P
import json
import os


# Load module specs
f = open('question-identifier-specs.json')
specs = json.load(f)

# Set enviroment configs up. Default values can be changed by altering
# the second argument in each "get" call
QUEUE_SERVER_HOST, QUEUE_SERVER_PORT = os.environ.get(
    "QUEUE_SERVER", "200.17.70.211:10163").split(":")

#Cria o objeto do M2P e passa o classificador
MyM2P = M2P(QUEUE_SERVER_HOST, QUEUE_SERVER_PORT, specs, __init__.main)

#Faz a conexão com o servidor do M2P
MyM2P.connect()

#Recebe a função e envia a informação que aparecerá no servidor M2P
MyM2P.run()