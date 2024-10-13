# Usando a imagem base do Python
FROM python:3.9-slim

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando o arquivo requirements.txt para o container
COPY . .

# Instalando as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiando o código da aplicação para o container
COPY . .

# Expondo a porta que o Flask usará
EXPOSE 5000

# Comando para executar a aplicação
CMD ["python3", "main.py"]
