from flask import Flask, render_template, send_file, abort
import os
import pandas as pd

diretorio = './dataset'

pastas = [nome for nome in os.listdir(diretorio) if os.path.isdir(os.path.join(diretorio, nome))]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", pastas=pastas)

@app.route('/datasets/<folder_number>/arquivos')
def datasetsFiles(folder_number):
    diretorio = f'./dataset/{folder_number}'
    
    files = [nome for nome in os.listdir(diretorio) if os.path.isfile(os.path.join(diretorio, nome))]
    return render_template("choose_file.html", files=files, folder_number=folder_number)

@app.route('/dataset/<folder_number>/<filename>')
def dataset(folder_number, filename):
    arquivo_parquet = f'./dataset/{folder_number}/{filename}'
    df = pd.read_parquet(arquivo_parquet)
    
    dados = df.to_dict(orient='list')
    
    return render_template('dataset.html', dados=dados, columns=df.columns.tolist(), rows_count=df.shape[0], filename=filename, folder_number=folder_number)

@app.route('/download/<folder_number>/<filename>')
def download_file(folder_number, filename):
    file_path = f'./dataset/{folder_number}/{filename}'
    
    if not os.path.isfile(file_path):
        return abort(404) 
    
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

