#!/bin/bash

conda update -n base -c conda-forge conda
echo "Verificando a instalação do Conda..."
source /opt/conda/etc/profile.d/conda.sh

if [ -d "/opt/conda/envs/3W" ]; then
    echo "Removendo o diretório do ambiente '3W' existente..."
    rm -rf /opt/conda/envs/3W
fi


echo "Criando o ambiente '3W'..."
conda env create -f environment.yml || { echo "Falha ao criar o ambiente '3W'"; exit 1; }

echo "Ativando o ambiente '3W'..."
source activate 3W || { echo "Falha ao ativar o ambiente '3W'"; exit 1; }

pip install pydantic --root-user-action=ignore

if ! command -v jupyter &> /dev/null; then
    echo "Jupyter Notebook não encontrado. Verifique o environment.yml."
    exit 1
else
    echo "Iniciando o Jupyter Notebook..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
fi
