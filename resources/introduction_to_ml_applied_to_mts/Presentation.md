# Guia de apresentação — Introduction to Machine Learning applied to Multivariate Time Series

Roteiro para quem for **apresentar ou ensinar** este recurso (oficina, minicurso ou aula). Ele não substitui o [README](README.md) nem os notebooks: organiza o que dizer, o que mostrar e o que fazer se algo der errado.

O fio condutor vem do livro-companheiro:

> Lopes et al. (2026). *Aprendizado de Mundo Aberto em Poços de Petróleo: da detecção baseada em regras à explicabilidade por agentes de linguagem.* ISBN 978-65-02-28836-8. DOI [10.5281/zenodo.22306796](https://doi.org/10.5281/zenodo.22306796). CC BY 4.0.

As referências "Cap. N" e "Ap. E" abaixo são capítulos e apêndices desse livro. As gravações do 4º Workshop on 3W (Partes 1 e 2) estão linkadas no [README](README.md#-workshop-presentation) e servem como referência de ritmo.

---

## 1. Mensagem central

Se a plateia levar só três ideias, que sejam estas (são as "hipóteses que a prática desmente" do livro, em escala de oficina):

1. **O número que você reporta depende de decisões que você tomou antes do modelo** — escalonamento, janelamento, de onde vêm os dados (real, simulado, desenhado), como se divide treino e teste. Cada decisão deste recurso é editável em `src/config.py`.
2. **Acurácia simples engana sob desbalanceamento**, e média esconde variabilidade (entre dobras, entre poços). Use acurácia balanceada e olhe a tabela completa, não só a média.
3. **Visualizar, agrupar e detectar novidade são ferramentas de inspeção, não de medida**: uma projeção t-SNE é uma escolha, o limiar de erro de reconstrução move o compromisso detecção × alarme falso mais que a arquitetura, e o K "ótimo" raramente é o número de classes.

Frase para abrir: *"Hoje vocês vão ver o mesmo conjunto de dados contar histórias diferentes dependendo de decisões que parecem detalhes."*

## 2. Público e pré-requisitos

- Python básico; noção de aprendizado de máquina; familiaridade com séries temporais ajuda. Conhecimento de petróleo e gás **não** é necessário (explique os sensores quando aparecerem).
- Não é preciso GPU. Com a configuração padrão, a execução completa leva cerca de 10–15 minutos em uma CPU de 8 *threads*.

## 3. Preparação

### Na véspera

- [ ] Clonar o repositório e criar o ambiente (instruções no [README](README.md#-usage-instructions)):
  ```bash
  uv venv .venv
  uv pip install -e "../../.[resources]"   # o extra "resources" instala o UMAP
  ```
- [ ] Conferir que o **3W Dataset 2.0.0** está em `../../dataset/` (cerca de 2.228 arquivos Parquet). Se faltar, `ParquetDataset` baixa do Figshare no primeiro uso — **não deixe isso para a hora da aula** (o download é grande).
- [ ] **Executar os cinco notebooks de ponta a ponta** (`1` primeiro; os demais em qualquer ordem) e conferir que não há erros. Isso também gera `processed_data/`, de que os notebooks 2–5 dependem.
- [ ] Guardar uma cópia dos notebooks **já executados** (com saídas): são o plano B se a máquina falhar.
- [ ] Ter o livro aberto (PDF) nos capítulos citados abaixo, para mostrar figuras e tabelas quando a pergunta surgir. Atenção: parte das figuras do livro vem de artigos com outras licenças; **mostre, não redistribua**.

### 30 minutos antes

- [ ] Abrir o Jupyter no kernel do ambiente criado e rodar `import src` na primeira célula do notebook 1 (confirma instalação do toolkit e do PyTorch).
- [ ] Confirmar que `processed_data/cv_splits/windowed/` existe (use o bloco *Quick Start* do README).
- [ ] Fechar aplicativos pesados: os notebooks 1 e 5 chegam a usar vários GB de RAM (`MAX_FILES_PER_CLASS = 50` ≈ 4–5 GB).

## 4. Roteiro e tempos

Sugestão para duas sessões (como no workshop original). Os tempos incluem conversa; a execução ao vivo é curta, mas **não execute tudo ao vivo** — mostre as saídas já geradas e rode só as células-chave.

| Bloco | Conteúdo | Tempo |
|---|---|---|
| **Sessão 1** | | **≈ 2 h** |
| Abertura | Mensagem central, o conjunto 3W, o livro, o que é e o que não é este recurso | 15 min |
| Notebook 1 | Tratamento de dados | 40 min |
| Notebook 2 | Técnicas de visualização | 25 min |
| Notebook 3 | Aprendizado não supervisionado (OCSVM e autoencoder LSTM) | 40 min |
| **Sessão 2** | | **≈ 1 h 45** |
| Notebook 4 | Aprendizado supervisionado | 40 min |
| Notebook 5 | Métodos de agrupamento | 35 min |
| Fechamento | Síntese, 3W Toolkit, exercícios, próximos passos | 20 min |
| Perguntas | | 10 min |

Versão curta (≈ 1 h): abertura (10 min) + Notebook 1 (15) + Notebook 3, só o autoencoder (20) + Notebook 4, só a comparação final (10) + fechamento (5).

---

## 5. Abertura (15 min)

**O que dizer**

- *O problema:* um poço submarino é um sistema que você não vê; sensores (pressão e temperatura no fundo, no TPT e no PDG) são a janela. Detectar eventos indesejáveis cedo evita parada e perda de produção.
- *O conjunto 3W:* série temporal **multivariada**, rotulada por instância, em três procedências — **real** (`WELL-*`), **simulada** (`SIMULATED-*`) e **desenhada** (`DRAWN-*`), com dez classes (0 = normal; 1–9 = tipos de evento). Mostre a pasta `dataset/` e o [README do dataset](../../dataset/README.md).
- *O livro:* conta a passagem de três hipóteses que a prática desmente — (1) o conjunto de classes é fechado; (2) detectar é o mesmo que localizar no tempo; (3) uma decisão correta sem justificativa é utilizável. Este recurso exercita a **Parte I** (fundamentos) e o Ap. E mapeia cada caderno a capítulos.
- *O que este recurso é:* um **roteiro de aprendizagem** com configurações reduzidas para caber numa aula. **Não** reproduz os experimentos do livro (esses estão no Ap. D). A tabela de diferenças está no [README](README.md#-companion-book).
- *O 3W Toolkit:* agora é o pacote oficial; o recurso usa o `ParquetDataset`, as métricas e o `CorrelationHeatmap` dele. Para um tutorial completo do toolkit, indique [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md).

**Pergunta para a plateia:** *"Quem já treinou um classificador em dados industriais e obteve 99% de acurácia?"* (Prepare o gancho para o Notebook 4.)

---

## 6. Notebook 1 — `1_data_treatment` (40 min)

**Objetivo:** ir dos arquivos Parquet brutos a janelas prontas para modelos, tomando decisões conscientes.
**Livro:** Cap. 3 (dados, armadilhas, pré-processamento) e Cap. 4 (validação). **Toolkit:** `ParquetDataset`.

**Roteiro**

1. **Carga e procedência** (célula de *Step 1*). Mostre a saída de `explore_structure()`: contagens de arquivos reais, simuladas e desenhadas por classe. *Mensagem (Cap. 3, "Três procedências"):* misturar procedências sem declarar é a forma mais comum de inflar resultados. O recurso carrega **primeiro os reais** e completa o limite por classe com `config.NON_REAL_EVENT_TYPES` (simulados **e** desenhados, por padrão). O livro é mais estrito (nunca usa desenhados; simulados só em classes escassas) — mostre como aproximar isso com `NON_REAL_EVENT_TYPES = ["simulated"]`.
2. **Análise dos dados brutos.** Mostre uma série típica. *Armadilhas do Cap. 3:* valores ausentes preenchidos por *forward/backward fill*; sensor **congelado** não vira `NaN` (é valor constante) e só se detecta olhando o desvio-padrão; a correlação entre sensores muda com o estado das válvulas (o 3W não distribui esse estado); e — **o achado mais incômodo** — em várias classes (2, 6, 7 e 8) o início da anomalia coincide com a perda dos medidores de fundo (`P-PDG`, `T-PDG`): um modelo pode aprender *"sensor de fundo sumiu → anomalia"*, regra que pontua bem e não serve na operação. Vale uma pausa: este recurso usa `P-PDG`.
3. **Escalonamento** (comparação dos quatro métodos). Min-Max onde a saída do autoencoder é sigmoide (Cap. 7); *z-score* onde o modelo supõe dados aproximadamente gaussianos (Mahalanobis, Cap. 12). *Seja honesto:* aqui cada evento é escalonado com **as próprias estatísticas** (`fit_transform` por dataframe) — simples, mas remove níveis absolutos; o livro ajusta o escalonador **só no treino**.
4. **Validação cruzada.** Política: treino com real + simulado (+ desenhado), teste só com real; dobras **por instância**, não por janela. *Cap. 3 e 4:* janelas sobrepostas de uma mesma instância em treino e teste vazam informação; janelas de um mesmo poço dos dois lados fazem o modelo aprender o poço. O livro prefere divisão temporal 80/20 ou avaliação por poço.
5. **Janelamento.** `WINDOW_SIZE = 300`, `WINDOW_STRIDE = 150`, `SAMPLING_RATE = 5`. *Ponto que quase ninguém percebe:* com `SAMPLING_RATE = 5` uma janela de 300 pontos cobre **1.500 s**, não 300 s — compare com o livro pelo **tempo coberto**, não pela contagem de pontos.

**Perguntas para a plateia**

- "Se eu trocar `SAMPLING_RATE` de 5 para 1, o que muda: a resolução, o tempo coberto, ou os dois?"
- "Por que particionar por arquivo e não por janela?"

**Pontos de atenção**

- A célula de carga é a mais pesada; se faltar memória, reduza `MAX_FILES_PER_CLASS` (é o primeiro parâmetro a ajustar).
- O notebook termina salvando `processed_data/`. Sem ele, os notebooks 2–5 falham.
- Se perguntarem "por que não usar `Normalize`, `Windowing` e `KFoldSplitter` do toolkit?": porque o resultado mudaria (veja a tabela do [README](README.md#-relationship-with-the-3w-toolkit)). `Normalize` usa normas L1/L2/máx., não Min-Max; `Windowing` preenche o início do sinal e devolve um dataframe com índice múltiplo; `KFoldSplitter` não conhece "treinar com simulado, testar só com real".

**Exercícios (Ap. E):** comparar os quatro escalonadores; mudar `SAMPLING_RATE`; implementar a política de procedência do livro.

---

## 7. Notebook 2 — `2_visualization_techniques` (25 min)

**Objetivo:** olhar as janelas em 2D com t-SNE e UMAP — e aprender a desconfiar do que se vê.
**Livro:** Cap. 6, §*Visualização*; Ap. E, §*Caderno 2*. **Toolkit:** não tem t-SNE/UMAP; para gráficos de séries 3W veja o [tutorial de visualização](../introduction_to_3w_toolkit_v2.0.0/6_data_visualization.ipynb).

**Roteiro**

1. Mostre as séries por classe (visualização completa): o que é "normal" e o que é "diferente".
2. Achatamento das janelas (300 × 3 = 900 dimensões), padronização e amostragem balanceada.
3. **Quatro configurações de t-SNE** (perplexidades 10, 30 e 50; taxas de aprendizado diferentes) e **quatro de UMAP** (5 a 50 vizinhos; métricas euclidiana, cosseno, Manhattan). Deixe a plateia comparar as figuras.
4. Comparação final dos métodos.

**A mensagem é a regra do Cap. 6:** *nenhuma conclusão quantitativa do livro se apoia em uma projeção 2D*. Em t-SNE, distâncias entre aglomerados **não têm significado**; o resultado depende da perplexidade e da semente. UMAP preserva mais estrutura global, mas compartilha a limitação. **A projeção é uma escolha, não uma medida.** Cuidado ao ler as frases de "estabilidade/determinismo" do UMAP no texto do notebook: valem para semente e parâmetros fixos.

**Pontos de atenção**

- Requer o extra `resources` (UMAP). Sem ele, o notebook continua sem a parte do UMAP e avisa como instalar.
- O aviso `n_jobs value 1 overridden to 1 by setting random_state` do UMAP é esperado (reprodutibilidade tem preço em paralelismo).

**Atividade rápida (3 min):** mude `random_state` em `run_tsne_analysis` (`src/visualization.py`) e veja que conclusões de separabilidade sobrevivem.

---

## 8. Notebook 3 — `3_introduction_to_unsupervised_learning` (40 min)

**Objetivo:** detectar anomalias treinando **só com dados normais** — primeiro com uma abordagem geométrica (OCSVM), depois com um autoencoder LSTM.
**Livro:** Cap. 5 (autoencoders), Cap. 6 (escores de novidade), Cap. 7 (sistema dual), Cap. 12 (Mahalanobis no espaço latente); Ap. E, §*Caderno 3*. **Toolkit:** o autoencoder é escrito em **PyTorch** (o mesmo framework do toolkit); para `MLP`, `ModelTrainer` e `ModelAssessment` veja o [tutorial 7](../introduction_to_3w_toolkit_v2.0.0/7_model_training_and_evaluation.ipynb).

**Roteiro**

1. **Avaliação por dobra.** Treina-se com os normais de cada dobra; testa-se com normais e anomalias (classes 3, 4 e 8) da mesma dobra.
2. **OCSVM.** A janela é achatada num vetor de 900 posições, padronizada e passada ao OCSVM. *Gancho com o livro (Cap. 8):* a mesma família de método obteve **0,65** de acurácia contra **0,87** do *ensemble* binário — e a razão aparece ao olhar a dimensão do vetor: em 900 dimensões as distâncias se concentram.
3. **Autoencoder LSTM** (`StableLSTMAutoencoder`). Cada escolha de estabilidade existe porque a alternativa falha (Ap. E): ativação `tanh`; inicialização ortogonal nos pesos recorrentes; corte de gradiente (`clipnorm = 1,0`); taxa de aprendizado `0,0005`; saída sigmoide (casa com o Min-Max); sementes fixas. *Uma perda que vira `NaN` no meio do treino costuma ser culpada nos dados — é sintoma dessas escolhas.* O sistema do Cap. 7, em operação, usa ReLU e funciona por ter um gargalo bem mais estreito (4 dimensões): **as escolhas interagem**.
4. **O limiar.** `AnomalyDetector` implementa o **percentil 95** dos erros normais e **μ + 2σ**. *Mensagem:* o limiar move o compromisso detecção × alarme falso mais que a arquitetura. Mostre a curva ROC e onde cada limiar cai.

**Perguntas para a plateia**

- "Se o erro de reconstrução de uma anomalia é alto, o que eu sei? E se for baixo?"
- "O que acontece com o limiar se eu treinar com normais de poços diferentes dos de teste?"

**Pontos de atenção**

- É o notebook mais demorado (≈ 3–6 min na CPU). Mostre as saídas prontas e rode apenas a célula de arquitetura/treino de uma dobra, se quiser demonstrar.
- Com sementes fixas os números se repetem entre execuções na mesma máquina e versões de pacotes; entre máquinas podem variar um pouco (ponto flutuante).
- Não compare os números daqui com os do livro: janelas, sensores, latente e partição são diferentes (tabela no README).

**Exercícios (Ap. E):** `SAMPLING_RATE` 5 → 1; percentil 95 × μ + 2σ no mesmo erro; Mahalanobis na penúltima camada de um classificador do Notebook 4.

---

## 9. Notebook 4 — `4_introduction_to_supervised_learning` (40 min)

**Objetivo:** treinar e comparar seis modelos clássicos e aprender a avaliá-los.
**Livro:** Cap. 4 (matriz de confusão, acurácia balanceada, validação) e Cap. 8; Ap. E, §*Caderno 4*. **Toolkit:** `ThreeWToolkit.metrics` (`accuracy_score`, `balanced_accuracy_score`); para `SklearnModels`, `MLP`, `ModelTrainer`, `ModelAssessment` e `Pipeline`, veja os tutoriais [7](../introduction_to_3w_toolkit_v2.0.0/7_model_training_and_evaluation.ipynb) e [8](../introduction_to_3w_toolkit_v2.0.0/8_pipeline_integration.ipynb).

**Roteiro**

1. Classes selecionadas por padrão: **{0, 3, 4, 8}** — normal, golfadas severas, instabilidade de fluxo e hidrato na linha de produção.
2. Modelos: árvore de decisão, floresta aleatória, SVM linear e RBF, e três redes densas (rasa, profunda, regularizada).
3. **Balancear o treino e não o teste.** A assimetria é deliberada: balancear o teste *muda a pergunta* (mede-se sob uma distribuição que não existe na operação). Para o desbalanceamento na avaliação, use acurácia balanceada.
4. **Acurácia × acurácia balanceada.** As tabelas por dobra mostram `Test Acc` e `Test Bal. Acc` lado a lado. *Cap. 4:* um detector inútil atinge 99,7% de acurácia.
5. **Variabilidade entre dobras.** Aponte para a tabela final: a diferença **entre dobras** costuma ser maior que a diferença **entre algoritmos**. É o argumento do livro a favor de avaliar poço a poço e reportar a tabela completa, não só a média.
6. **A classe 4** aparece entre as selecionadas: é a classe sintomática do fundo de todas as tabelas do livro (argumento de taxonomia do Cap. 16).

**Pergunta para a plateia:** *"Qual modelo é o melhor?"* — e só depois mostrar a tabela com desvio e consistência.

**Pontos de atenção**

- As árvores e a floresta têm sobreajuste visível (treino ≈ 0,99; teste bem menor em algumas dobras). É um bom momento para falar de generalização entre poços.
- Notebook ≈ 2–3 min na CPU.

**Exercícios (Ap. E):** política de procedência do livro; `selected_classes = None` e comparação das classes 4 e 5.

---

## 10. Notebook 5 — `5_clustering_methods` (35 min)

**Objetivo:** descobrir estrutura sem rótulos e medir se ela faz sentido.
**Livro:** Cap. 6 (§*Agrupamento* e §*Avaliar agrupamentos sem rótulos*) e Cap. 9 (ciclo de mundo aberto); Ap. E, §*Caderno 5*. **Toolkit:** o sub-pacote `ThreeWToolkit.clustering` faz agrupamento **por instância** (DTW, hierárquico); veja as demos [clustering_pipeline_examples](../../toolkit/demos/clustering_pipeline_examples.ipynb) e [clustering_real_vs_simulated_examples](../../toolkit/demos/clustering_real_vs_simulated_examples.ipynb).

**Roteiro**

1. **Dados completos redimensionados para 500 pontos** (diferente dos notebooks anteriores, que usam janelas), PCA para visualização.
2. **K-means:** cotovelo e silhueta para escolher K. *Cap. 9:* em mundo aberto, **K é exatamente o que não se conhece** — o livro treina uma floresta aleatória para *prever* K a partir de oito características do próprio conjunto. Aqui você verá o outro lado: o K sugerido pelos índices raramente coincide com o número de classes.
3. **MeanShift** (K emerge dos modos; exige a largura de banda) e **DBSCAN** (único que rotula ruído; em alta dimensão tende a rotular quase tudo como ruído — por isso o livro só o usa como comparação).
4. **Avaliação:** silhueta (interna) e índice de Rand ajustado e acurácia com atribuição húngara (externas).
5. **Mensagem do Ap. E:** este notebook é o ciclo de mundo aberto **sem a etapa de rejeição**; você encontra, em minutos, os dois achados do Cap. 9 — a silhueta é baixa no espaço bruto e o número de grupos sugerido não coincide com o de classes. O Cap. 12 explica o porquê; **rodar antes de ler torna a explicação mais convincente**.

**Pergunta para a plateia:** *"O DBSCAN rotulou grande parte dos pontos como ruído. O algoritmo falhou ou o espaço é inadequado?"*

**Pontos de atenção**

- Rápido (< 1 min). Algumas mensagens de "baixa concordância com a verdade" são **esperadas** — são o objetivo didático.

**Exercício (Ap. E):** trocar o espaço de entrada pelo **espaço latente do autoencoder** do Notebook 3 e recalcular silhueta e Rand ajustado.

---

## 11. Fechamento (20 min)

1. **Retomar as três mensagens centrais** com um exemplo de cada notebook: procedência/amostragem (1), projeção como escolha (2), limiar (3), acurácia balanceada e variabilidade entre dobras (4), K e silhueta (5).
2. **De onde vem a ponte para o livro:** Parte I (fundamentos) = o que fizemos hoje; Parte II (Caps. 7–9) mostra o sistema dual, o *ensemble* binário e o aprendizado de mundo aberto; Parte III, a classificação em mundo fechado, a segmentação com U-Net e Mahalanobis no espaço latente; Parte IV, a explicabilidade por modelos de linguagem; Parte V, a síntese e os requisitos de operação.
3. **Próximos passos:** (a) o tutorial [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md) para construir *pipelines* completos; (b) os exercícios guiados do Ap. E (todos os cadernos terminam com os deles); (c) citar o livro e o 3W.
4. **Convite:** contribuições e dúvidas pelo repositório do projeto ([petrobras/3W](https://github.com/petrobras/3W)).

---

## 12. Plano B (se algo falhar)

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `ModuleNotFoundError: ThreeWToolkit` | Ambiente sem o toolkit instalado | `uv pip install -e "../../.[resources]"` no ambiente do kernel; reiniciar o kernel |
| `ModuleNotFoundError: umap` | Extra `resources` ausente | Instalar o extra; ou pular a seção de UMAP do Notebook 2 |
| Notebook 1 trava ou estoura memória | `MAX_FILES_PER_CLASS` alto para a máquina | Reduzir `MAX_FILES_PER_CLASS` (ex.: 20) e reexecutar |
| Dataset ausente / download lento | `../../dataset/` vazio | Baixar antes (Figshare, versão 2.0.0); na aula, usar a cópia pré-executada |
| Notebooks 2–5 dizem "execute o notebook 1" | `processed_data/` ausente | Rodar o Notebook 1; ou restaurar a pasta `processed_data/` da véspera |
| Treino do autoencoder lento | CPU fraca | Reduzir `epochs` na célula de treino (padrão 20) ou mostrar as saídas salvas |
| Resultados diferentes dos da tela | Versões/máquina diferentes | Esperado em pequena escala; reforce que os **padrões** (variabilidade entre dobras, efeito do limiar), não os dígitos, são o ponto |
| Figuras com caracteres faltando | Fonte sem emoji | Cosmético; ignorar |

## 13. Perguntas frequentes (e respostas curtas)

- **"Por que não usar todas as classes?"** Para caber no tempo e para a discussão ficar nítida; `selected_classes = None` libera todas (exercício do Ap. E).
- **"Por que o treino inclui simulados e desenhados?"** Maximiza volume de treino e mantém a avaliação honesta (teste só com real). O livro adota outra política (simulados só nas classes escassas; desenhados nunca). Ambas são defensáveis; o que não é defensável é misturar sem declarar. Comparar as duas é o exercício 3 do Ap. E.
- **"Por que o OCSVM foi melhor que o autoencoder aqui?"** Cuidado ao generalizar: configurações reduzidas, três dobras, limiar por percentil 95. O livro discute que o limiar e o espaço onde se mede a novidade pesam mais que a escolha do modelo (Caps. 6 e 12).
- **"Posso usar isto com outro conjunto de dados?"** Sim, a parametrização está em `src/config.py`; o pipeline é configurável (sensores, janela, escalonamento). Para conjuntos fora do 3W, adapte `DataLoader`.
- **"O toolkit já faz tudo isto?"** Parte: carga, métricas, visualização, *pipelines* e agrupamento por instância. O recurso mantém localmente o que o toolkit não oferece ou faria de outra forma (README, seção *Relationship with the 3W Toolkit*).
- **"Onde está o código do autoencoder do livro?"** Este recurso traz uma versão didática em PyTorch; os experimentos do livro estão descritos no Ap. D.

## 14. Changelog: o que do 3W Toolkit foi incorporado

Resumo, para quem já conhecia a versão anterior deste recurso, do que passou a vir do [3W Toolkit](../../toolkit/ThreeWToolkit/README.md) (agora o pacote oficial) e do que continua local. Regra seguida: **só se substituiu o que não altera os resultados**; o que mudaria foi mantido (e explicado no [README](README.md#-relationship-with-the-3w-toolkit)).

### Acrescentado a partir do toolkit

| Componente do toolkit | Onde entra | O que muda para quem apresenta |
|---|---|---|
| `dataset.ParquetDataset` / `ParquetDatasetConfig` | `src/data_loader.py` (`DataLoader`) | A descoberta e a leitura dos arquivos passam pelo toolkit. Os arquivos carregados e os DataFrames são os mesmos de antes. A procedência (**real**, **simulada**, **desenhada**) agora é explícita na saída do Notebook 1 (antes, desenhados eram contados como simulados). Se o dataset 2.0.0 faltar em `../../dataset/`, o toolkit o baixa do Figshare. |
| `metrics.accuracy_score` | `src/supervised_classification.py` | Mesmos valores de acurácia de antes (o toolkit usa o scikit-learn por baixo). |
| `metrics.balanced_accuracy_score` | `src/supervised_classification.py` | **Novo:** as tabelas por dobra do Notebook 4 ganharam a coluna `Test Bal. Acc` — base para a discussão de acurácia × acurácia balanceada (Cap. 4). |
| `data_visualization.CorrelationHeatmap` | `src/visualization.py` (`plot_correlation_matrix`) | Mesmo gráfico, agora feito pelo toolkit. |
| Dependências e ambiente | `pyproject.toml` do repositório | O recurso usa o toolkit instalado (`uv pip install -e "../../.[resources]"`). O extra `resources` instala o UMAP. **TensorFlow não é mais necessário.** |
| Tutoriais do toolkit | Notebooks 1, 2, 3, 4 e 5 | Cada notebook aponta para o tutorial ou demo correspondente do toolkit (`ParquetDataset`, visualização, treino e avaliação, *pipelines*, agrupamento por instância). |

Efeitos colaterais de usar o toolkit: ao ser importado, ele configura o log em nível INFO; por isso `src/__init__.py` rebaixa o log raiz para WARNING, para as saídas dos notebooks ficarem legíveis.

### Avaliado e mantido local (e por quê)

| Local | Equivalente no toolkit | Motivo |
|---|---|---|
| Escalonamento (`minmax`, `standard`, `robust`, `normalizer`) | `preprocessing.Normalize` | O toolkit usa normas L1/L2/máx.; a transformação é outra. |
| `create_time_windows` | `feature_extraction.Windowing` | O toolkit preenche o início do sinal, usa sobreposição fracionária e devolve DataFrame com índice múltiplo; o recurso devolve um DataFrame por janela com metadados e aceita janela mínima menor que a janela. |
| Preenchimento de ausentes (*ffill* + *bfill*) | `preprocessing.ImputeMissing` | O toolkit dá erro se um sensor estiver inteiramente ausente num evento, o que é frequente no 3W. |
| `CrossValidator` | `utils.KFoldSplitter` | O divisor do toolkit trabalha com `BaseDataset` e não conhece "treinar com simulado, testar só com real". |
| Redimensionamento para 500 pontos | `clustering.TimeSeriesResampler` | Método diferente (fatiar/média por passo). |
| t-SNE/UMAP, K-means/MeanShift/DBSCAN, aumento de dados, OCSVM | — | O toolkit não oferece. |
| Autoencoder LSTM | `models.MLP`, `trainer.TorchTrainer` | O toolkit não tem autoencoder recorrente. O modelo agora é escrito em **PyTorch**, o framework do toolkit. |

### Outras mudanças que afetam a apresentação

- **Autoencoder em PyTorch** (antes TensorFlow/Keras): mesma API e mesmas decisões de estabilidade; os números do Notebook 3 mudaram em relação às gravações do workshop.
- **Reprodutibilidade:** a amostragem de janelas do Notebook 3 agora usa semente fixa; execuções repetidas dão os mesmos números.
- **Correções:** `boxplot(labels=...)` (quebrava com o matplotlib atual), gráfico do OCSVM (acurácias complementares e "ROC invertido"), chamadas de gráfico de K-means com assinatura antiga (avisos no Notebook 5) e instalação de pacotes via `subprocess` no Notebook 2.
- **Conteúdo:** cada notebook ganhou referências ao livro (capítulos e seções) e termina com os exercícios do Ap. E; o README ganhou a citação do livro e a tabela de diferenças em relação aos experimentos dele.

---

## 15. Referências

- **Livro:** Lopes, L. G. O. et al. (2026). *Aprendizado de Mundo Aberto em Poços de Petróleo.* DOI [10.5281/zenodo.22306796](https://doi.org/10.5281/zenodo.22306796). Cap. 3 (dados), 4 (avaliação), 5 (aprendizado profundo), 6 (novidade e agrupamento), 7 (sistema dual), 8 (ensemble binário), 9 (mundo aberto), 12 (Mahalanobis), 16 (síntese); Ap. D (reprodução) e Ap. E (prática).
- **3W Project:** [github.com/petrobras/3W](https://github.com/petrobras/3W); [3W Dataset](../../dataset/README.md); [3W Toolkit](../../toolkit/ThreeWToolkit/README.md); [3W Dataset no Figshare](https://figshare.com/projects/3W_Dataset/251195).
- **Tutoriais do toolkit:** [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md).
- **Este recurso:** [README](README.md).
