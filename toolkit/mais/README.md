# Sistema MAIS

Fault detection of oil wells using machine learning.

Amostras de experimentos realizáveis estão no diretório experiments/.

De forma geral, o processo pra treinar os modelos é o seguinte:
Dentro do diretório do experimento, por ex., 'experiments/multiclass/'

  1. [OPCIONAL] Inicializar o servidor do mlflow com URI sqlite para os logs. De forma geral, essa opção é melhor para
  evitar que o número de arquivos gerados pelo mlflow seja alto.

  2. Executar  'tune\_lgbm.py'. Esse script possui múltiplos comandos que podem ser listados com o comando '--help'.
  [NOTA: Use as variáveis de ambiente apropriadas para o seu sistema de log do mlflow.]

Classes utilitárias:

  1. mais/data/dataset.py -- MAEDataset tem o núcleo da lógica de ler cada csv, passar pra extração de atributos
    e entregar as tabelas completas no final.
  2. mais/data/feature\_mappers.py -- tem classes que realizam a extração de atributos. As instâncias dessas classes
    são passadas nas funções do MAEDataset, normalmente através de um experimento.
    TorchStatisticalFeatureMapper faz estatísticas de janela retangular, TorchWaveletFeatureMapper faz wavelets e
    TorchEWStatisticalFeatureMapper faz estatísticas em janela exponencial.
  3. src/data/label\_mappers.py tem classes que definem o modo de detecção, tipo transiente ou não, multiclasse,
  valor mais comum ou último valor.
