# Guia de apresentação — Introduction to Machine Learning using the 3W Toolkit 3.2.1

Roteiro para quem for **apresentar ou ensinar** este recurso (**Introduction to Machine Learning using the 3W Toolkit 3.2.1**, que apresenta os principais conceitos de ML com exemplos de uso do toolkit) em **duas aulas, como na programação do evento: Parte 1 de 2, com 35 min (dados, pré-processamento e visualização: notebooks 1 e 2), e Parte 2 de 2, com 1 h (classificação, anomalias e agrupamento: notebooks 4, 3 e 5)**, no workshop do 3W (dias 18 e 19/11/26). Ele não substitui o [README](README.md) nem os notebooks: organiza o que dizer, o que mostrar, o que rodar ao vivo, o que cortar se faltar tempo e o que fazer se algo der errado.

O fio condutor vem do livro-companheiro:

> Lopes et al. (2026). *Aprendizado de Mundo Aberto em Poços de Petróleo: da detecção baseada em regras à explicabilidade por agentes de linguagem.* ISBN 978-65-02-28836-8. DOI [10.5281/zenodo.22306796](https://doi.org/10.5281/zenodo.22306796). CC BY 4.0.

As referências "Cap. N" e "Ap. E" abaixo são capítulos e apêndices desse livro. As gravações do 4º Workshop on 3W (Partes 1 e 2) estão linkadas no [README](README.md#-classes-and-previous-material); elas cobrem a **versão anterior** do material (antes do toolkit 3.2.1), então servem só como referência de ritmo.

---

## 1. Mensagem central

Se a plateia levar só três ideias, que sejam estas (são as "hipóteses que a prática desmente" do livro, em escala de oficina):

1. **O número que você reporta depende de decisões que você tomou antes do modelo** — escalonamento, janelamento, de onde vêm os dados (real, simulado, desenhado), como se divide treino e teste. Cada decisão deste recurso é editável em `src/config.py`.
2. **Acurácia simples engana sob desbalanceamento**, e média esconde variabilidade (entre dobras, entre poços). Use acurácia balanceada e olhe a tabela completa, não só a média.
3. **Visualizar, agrupar e detectar novidade são ferramentas de inspeção, não de medida**: uma projeção t-SNE é uma escolha, o limiar de erro de reconstrução move o compromisso detecção × alarme falso mais que a arquitetura, e o K "ótimo" raramente é o número de classes.

**Onde cada mensagem aparece:** a 1 fecha a **Aula 1** (Notebook 1: escalonamento, dobras e janelas), junto com a primeira metade da 3 (Notebook 2: a projeção é uma escolha); a 2 aparece na **Aula 2** com o Notebook 4 (acurácia balanceada e a impressão digital do poço); a 3 fecha a **Aula 2** (Notebooks 3 e 5).

Frase para abrir: *"Hoje vocês vão ver o mesmo conjunto de dados contar histórias diferentes dependendo de decisões que parecem detalhes."*

## 2. Público e pré-requisitos

- Python básico; noção de aprendizado de máquina; familiaridade com séries temporais ajuda. Conhecimento de petróleo e gás **não** é necessário (explique os sensores quando aparecerem).
- Não é preciso GPU. Com a configuração padrão, a execução completa leva cerca de 10 minutos em uma CPU de 8 *threads*.

## 3. Preparação

### Na véspera

- [ ] Clonar o repositório e criar o ambiente (instruções no [README](README.md#-usage-instructions)):
  ```bash
  cd resources/introduction_to_ml_using_3w_toolkit_v3.2.1   # a partir da raiz do repositório
  uv venv .venv
  uv pip install -e "../../.[notebooks,resources]" jupyter   # "resources" instala o UMAP; "notebooks" o kernel e o nbconvert
  ```
- [ ] Conferir que o **3W Dataset 2.0.0** está em `../../dataset/` (cerca de 2.228 arquivos Parquet). Se faltar, `ParquetDataset` baixa do Figshare no primeiro uso — **não deixe isso para a hora da aula** (o download é grande).
- [ ] **Executar os cinco notebooks de ponta a ponta** (`1` primeiro; os demais em qualquer ordem) e conferir que não há erros. Isso também gera `processed_data/`, de que os notebooks 2–5 dependem.
- [ ] Guardar uma cópia dos notebooks **já executados** (com saídas): são o plano B se a máquina falhar. Nas duas aulas as saídas gravadas são o material principal; o ao vivo se limita às células leves marcadas na seção 4.
- [ ] **Os Notebooks 2 a 5 dependem do Notebook 1** (ele grava `processed_data/`). As duas aulas são em dias diferentes (18 e 19/11): execute o Notebook 1 antes da Aula 1 e mantenha a pasta `processed_data/` para a Aula 2.
- [ ] Ter o livro aberto (PDF) nos capítulos citados abaixo, para mostrar figuras e tabelas quando a pergunta surgir. Atenção: parte das figuras do livro vem de artigos com outras licenças; **mostre, não redistribua**.

### 30 minutos antes

- [ ] Abrir o Jupyter no kernel do ambiente criado e rodar `import src` na primeira célula do notebook 1 (confirma instalação do toolkit e do PyTorch).
- [ ] Confirmar que `processed_data/cv_splits/windowed/` existe (use o bloco *Quick Start* do README).
- [ ] Fechar aplicativos pesados: os notebooks 1, 4 e 5 chegam a usar 2 a 2,5 GB de RAM (`MAX_FILES_PER_CLASS = 50`).
- [ ] **Forma da gravação:** tela cheia (F11), sem a barra de favoritos e sem abas pessoais; deixe **abertas, em abas, as células-alvo** de cada notebook (não use a busca do navegador ao vivo, nem procure arquivos do `src/` durante a aula).

## 4. Roteiro das duas aulas (35 min + 1 h)

**Contexto do workshop.** As aulas fazem parte da programação do workshop do 3W. Antes delas, o **Panorama sobre o Projeto 3W** (Ricardo Vargas, dia 1) e o **minicurso do 3W Toolkit** (Bruno Martins, dias 1 e 2) já terão apresentado o projeto e o toolkit. Aqui o toolkit não é reapresentado: ele é **usado**, chamado direto nas células, e o que o toolkit não tem entra depois, como extensão. A Aula 1 (Parte 1 de 2) é no dia 18/11, às 09:25, com 35 min; a Aula 2 (Parte 2 de 2) é no dia 19/11, às 09:25, com 1 h.

Em cada notebook o **toolkit vem primeiro**; o que fazemos além dele (escalonadores extras, t-SNE/UMAP, autoencoder, K-means/DBSCAN) entra depois, como variação. **Não execute tudo ao vivo:** mostre as saídas gravadas e rode só as células marcadas com *(ao vivo)*, que são leves (o Notebook 1 inteiro roda em cerca de 1 min e o Notebook 5, em menos de 30 s).

### Aula 1 (Parte 1 de 2, 35 min): dados, pré-processamento e visualização

| Min | Bloco | Onde (seção do notebook) | Mostrar e rodar |
|---|---|---|---|
| 0–3 | Problema 3W e o que vamos fazer; o toolkit já foi apresentado | Seção 5 deste guia | Pasta `dataset/`; uma frase sobre `Config.build()` |
| 3–10 | Abrir e olhar os eventos | Notebook 1: *Toolkit first: `ParquetDataset`*; *Raw Data Analysis* | Célula *Toolkit first* (ao vivo); a célula que escolhe os eventos (só a saída: procedência); uma série bruta e a armadilha do `P-PDG` |
| 10–22 | Pré-processar | Notebook 1: *Scaling Methods Comparison*; *Choose the Scaler*; *Toolkit only: K-fold*; *Toolkit only: Windowing* | K-fold e `Windowing` do toolkit (ao vivo); comparação dos escalonadores e a montagem das dobras com os simulados, só as saídas |
| 22–31 | Ver as janelas em 2D | Notebook 2: *Toolkit first: a feature view of the windows*; *t-SNE Implementation*; *Method Comparison Analysis* | `StatisticalFeatures` × janela achatada (ao vivo); as quatro configurações de t-SNE (saída); a regra "projeção é escolha" |
| 31–35 | Fechamento da Aula 1 | Seção 11 | Mensagem 1 e primeira metade da 3; gancho para a Aula 2 |

**Se atrasar, corte nesta ordem:** (1) o UMAP e as figuras por classe do Notebook 2 (só cite que existem); (2) o detalhe da montagem das dobras (a política real + simulado vira uma frase); (3) a comparação dos escalonadores (só a saída); (4) a análise dos dados brutos (só a armadilha do `P-PDG`). **Não corte** a célula *Toolkit first*, o `KFoldSplitter`, o `Windowing` nem a regra "projeção é escolha": são o que a Aula 1 quer deixar.

### Aula 2 (Parte 2 de 2, 1 h): classificação, anomalias e agrupamento

**Abertura (30 s).** *"Classificação é quase toda do toolkit, agrupamento abre com o toolkit, e detecção de anomalias é o que o toolkit ainda não tem. Por isso o Notebook 3 é o que mais usa o `src/`."* O que vem do toolkit e o que não vem:

| Notebook | Do toolkit | Fora do toolkit (e por quê) |
|---|---|---|
| 4 · Classificação | `SklearnModelsConfig` (os sete modelos), `metrics.*` (acurácia, acurácia balanceada, precisão, revocação, F1) e `AssessmentVisualization` (matriz de confusão) | Achatar as janelas, balancear o treino e as tabelas e painéis de comparação (`src/`): os *trainers* do toolkit trabalham com eventos, não com janelas achatadas. O experimento do poço usa `GroupKFold` do scikit-learn |
| 3 · Anomalias | Só as métricas: `accuracy_score` e `roc_auc_score` | O OCSVM vem direto do scikit-learn (o `SklearnModelsConfig` o **recusa**: falta o método `score`); o autoencoder LSTM está em PyTorch no `src/` (o toolkit não tem autoencoder recorrente nem detectores de anomalia); limiares e curva ROC também ficam fora |
| 5 · Agrupamento | O bloco inicial inteiro: `InstanceQualityFilter`, `TimeSeriesResampler`, `TimeSeriesScaler`, `DistanceComputer` (DTW), `HierarchicalClusterer`, `DendrogramPlot` e o `ParquetDataset` | K-means, MeanShift, DBSCAN, silhueta, Rand ajustado e acurácia húngara (scikit-learn e `src/`): o agrupamento do toolkit é por DTW em eventos, com outro objetivo |

**Ordem: Notebook 4, depois 3, depois 5** (a tabela abaixo já segue essa ordem). O motivo: o aviso sobre a AUC do Notebook 3 depende da impressão digital do poço, que o Notebook 4 mostra. Se inverter, avise a plateia de que a explicação da AUC fica para o fim.

| Min | Bloco | Onde (seção do notebook) | Mostrar e rodar |
|---|---|---|---|
| 0–3 | Retomada da Aula 1 e gancho: "agora vamos treinar um modelo e ver o que o número esconde" | Seção 11 | Uma frase sobre a mensagem 1 |
| 3–23 | Classificar e avaliar | Notebook 4: *Toolkit first: one model, end to end*; tabelas por dobra; *Does the sensor level identify the well?* | Modelo de ponta a ponta (ao vivo, 8 min); acurácia × acurácia balanceada e variabilidade entre dobras (6 min); impressão digital do poço (6 min, saída gravada) |
| 23–38 | Detectar anomalias | Notebook 3: *Distance-Based Anomaly Detection*; *Building and Training the Autoencoder*; *ROC Curve Analysis*; aviso sobre a AUC | Só saídas (treino é de 3–6 min); OCSVM (5 min), autoencoder e limiar (7 min), AUC e o aviso do nível do sensor (3 min) |
| 38–54 | Agrupar | Notebook 5: *Toolkit first: hierarchical DTW clustering of events*; K-means; MeanShift e DBSCAN | DTW + hierárquico do toolkit (ao vivo, 5 min); K-means, silhueta e Rand ajustado (8 min); MeanShift e DBSCAN em 3 min |
| 54–60 | Discussão e fechamento da Aula 2 | Seção 11 | Mensagens 2 e 3; ponte para o livro; exercícios 3, 5 e 6 |

**Se atrasar, corte nesta ordem:** (1) MeanShift e DBSCAN (uma frase: os dois acham grupos compactos que não coincidem com as classes, com Rand ajustado em torno de 0,5); (2) a curva ROC (fique na AUC e no aviso); (3) a tabela por dobra de cada família do Notebook 4 (fique na tabela final). **Não corte** o modelo de ponta a ponta nem a impressão digital do poço (Notebook 4), o aviso sobre o nível do sensor (Notebook 3) nem o contraste silhueta × Rand ajustado do Notebook 5.

**Ritmo (ensaio da Aula 2).** O Notebook 3 consumiu o tempo do 5:

| Bloco | Planejado | No ensaio |
|---|---|---|
| Notebook 4 | 20 min | ≈ 17 min |
| Notebook 3 | 15 min | ≈ 27 min |
| Notebook 5 | 16 min | ≈ 5 min (cortado no K-means) |
| Fechamento | 6 min | ausente |

Para recuperar o tempo no Notebook 3, corte: a explicação longa da fronteira do OCSVM (≈ 4 min, com repetições), a arquitetura do autoencoder no `src/` (deixe só a saída do treino) e a comparação dos três limiares (fique no percentil 95). **Reserve os 6 min do fechamento** (seção 11): Mensagens 2 e 3, a ponte para o livro e os exercícios 3, 5 e 6.

**Oficina de dia inteiro:** os tempos completos por notebook são Notebook 1 (40 min), 2 (25), 3 (40), 4 (40) e 5 (35), mais abertura (15) e fechamento (20). As seções 6 a 10 trazem o roteiro completo de cada um, e o bloco *Na aula* de cada seção diz o que entra nas duas aulas.

---

## 5. Abertura da Aula 1 (3 min)

**Em 3 minutos:** o projeto e o toolkit já foram apresentados nos dias anteriores do workshop, então diga o problema em uma frase, mostre a pasta `dataset/`, lembre em uma frase o padrão `Config.build()` do toolkit e vá para o Notebook 1. O restante desta seção é apoio para perguntas.

**O que dizer**

- *O problema:* um poço submarino é um sistema que você não vê; sensores (pressão e temperatura no fundo, no TPT e no PDG) são a janela. Detectar eventos indesejáveis cedo evita parada e perda de produção.
- *O conjunto 3W:* série temporal **multivariada**, rotulada por instância, em três procedências — **real** (`WELL-*`), **simulada** (`SIMULATED-*`) e **desenhada** (`DRAWN-*`), com dez classes (0 = normal; 1–9 = tipos de evento). Mostre a pasta `dataset/` e o [README do dataset](../../dataset/README.md).
- *O livro:* conta a passagem de três hipóteses que a prática desmente — (1) o conjunto de classes é fechado; (2) detectar é o mesmo que localizar no tempo; (3) uma decisão correta sem justificativa é utilizável. Este recurso exercita a **Parte I** (fundamentos) e o Ap. E mapeia cada caderno a capítulos.
- *O que este recurso é:* um **roteiro de aprendizagem** com configurações reduzidas para caber numa aula. **Não** reproduz os experimentos do livro (esses estão no Ap. D). A tabela de diferenças está no [README](README.md#-companion-book).
- *O 3W Toolkit:* o recurso usa o toolkit **3.2.1**, chamado direto nas células (`ParquetDataset`, `Normalize`, `KFoldSplitter`, `Windowing`, `SklearnModelsConfig`, as métricas e o `clustering`); o `src/` guarda só o que o toolkit não tem. O toolkit é apresentado antes, no minicurso do workshop. Para um tutorial completo, indique [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md) (escrito para o toolkit 2.0.0) e as demos em [`../../toolkit/demos/`](../../toolkit/demos/) (linha 3.x).

**Pergunta para a plateia:** *"Quem já treinou um classificador em dados industriais e obteve 99% de acurácia?"* (Prepare o gancho para o Notebook 4.)

---

## 6. Notebook 1 — `1_data_treatment` (Aula 1, 19 min)

**Na aula (19 min).** *3–10:* célula *Toolkit first* (ao vivo), a célula que escolhe os eventos (só a saída: procedência) e uma série bruta com a armadilha do `P-PDG`. *10–22:* o `Normalize` e a comparação dos escalonadores (só a saída; o ponto é que o escalonamento vem **depois** de montar as dobras e é ajustado só no treino), o `KFoldSplitter` sozinho e o `Windowing` (ambos ao vivo). **Pule:** a teoria de escalonamento (leia em 1 min), a persistência dos dados e o resumo final. Os passos 1 a 5 do roteiro abaixo são o material completo.

**Objetivo:** ir dos arquivos Parquet brutos a janelas prontas para modelos, tomando decisões conscientes.
**Livro:** Cap. 3 (dados, armadilhas, pré-processamento) e Cap. 4 (validação). **Toolkit:** `ParquetDataset`, `PlotMultipleSeries`, `CorrelationHeatmap`, `ImputeMissing`, `Normalize`, `KFoldSplitter`, `Windowing`.

**Roteiro** (o toolkit vem primeiro em cada passo)

1. **Carga e procedência** (comece pela célula *Toolkit first*, que abre o `ParquetDataset` e plota um evento; depois a célula de *Step 1*, que escolhe os eventos com o mesmo `ParquetDataset`, até `MAX_FILES_PER_CLASS` eventos utilizáveis por classe: um evento com um sensor inteiramente vazio é descartado e o seguinte entra no lugar). Mostre a tabela de contagens: arquivos reais, simulados e desenhados, disponíveis e carregados por classe; a classe 6 fica com 6 eventos porque nenhum dos seus 215 simulados/desenhados tem o sensor `T-TPT`, e as classes 2 e 7 ficam abaixo de 50 porque não têm mais eventos utilizáveis (nenhuma delas entra nos cadernos de ML além do 3). O toolkit não limita por classe nem subamostra, então `clean_event` (`src/events.py`) faz a subamostragem por `SAMPLING_RATE`. *Mensagem (Cap. 3, "Três procedências"):* misturar procedências sem declarar é a forma mais comum de inflar resultados. O recurso carrega **primeiro os reais** e completa o limite por classe com `config.NON_REAL_EVENT_TYPES` (simulados **e** desenhados, por padrão). O livro é mais estrito (nunca usa desenhados; simulados só em classes escassas) — mostre como aproximar isso com `NON_REAL_EVENT_TYPES = ["simulated"]`.
2. **Análise dos dados brutos.** Mostre uma série típica. *Armadilhas do Cap. 3:* valores ausentes preenchidos por *forward/backward fill*; sensor **congelado** não vira `NaN` (é valor constante) e só se detecta olhando o desvio-padrão; a correlação entre sensores muda com o estado das válvulas (o 3W não distribui esse estado); e — **o achado mais incômodo** — em várias classes (2, 6, 7 e 8) o início da anomalia coincide com a perda dos medidores de fundo (`P-PDG`, `T-PDG`): um modelo pode aprender *"sensor de fundo sumiu → anomalia"*, regra que pontua bem e não serve na operação. Vale uma pausa: este recurso usa `P-PDG`.
3. **Escalonamento** (comparação dos quatro métodos: o `Normalize` do toolkit aparece primeiro, depois `standard`, `minmax` e `robust`). Min-Max onde a saída do autoencoder é sigmoide (Cap. 7); *z-score* onde o modelo supõe dados aproximadamente gaussianos (Mahalanobis, Cap. 12). O padrão agora é o `Normalize` do toolkit, **ajustado só nos eventos de treino de cada dobra** e aplicado ao teste, como o livro recomenda; os escalonadores do scikit-learn continuam sendo ajustados por evento (`fit_transform` por dataframe), o que remove níveis absolutos. O escalonamento acontece **depois** de montar as dobras. O `ImputeMissing` do toolkit preenche ausentes antes.
4. **Validação cruzada.** O `KFoldSplitter` do toolkit divide os eventos reais (estratificado por classe), e o próprio notebook acrescenta os simulados ao treino de cada dobra. Política: treino com real + simulado (+ desenhado), teste só com real; dobras **por instância**, não por janela. *Cap. 3 e 4:* janelas sobrepostas de uma mesma instância em treino e teste vazam informação; janelas de um mesmo poço dos dois lados fazem o modelo aprender o poço. O livro prefere divisão temporal 80/20 ou avaliação por poço.
5. **Janelamento** com o `Windowing` do toolkit (a célula de demonstração mostra o formato nativo, com índice `(window, variable)`). `WINDOW_SIZE = 300`, `WINDOW_OVERLAP = 0.5` (o `overlap` do `Windowing`), `SAMPLING_RATE = 5`. *Ponto que quase ninguém percebe:* com `SAMPLING_RATE = 5` uma janela de 300 pontos cobre **1.500 s**, não 300 s — compare com o livro pelo **tempo coberto**, não pela contagem de pontos.

**Perguntas para a plateia**

- "Se eu trocar `SAMPLING_RATE` de 5 para 1, o que muda: a resolução, o tempo coberto, ou os dois?"
- "Por que particionar por arquivo e não por janela?"

**Pontos de atenção**

- A célula de carga é a mais pesada; se faltar memória, reduza `MAX_FILES_PER_CLASS` (é o primeiro parâmetro a ajustar).
- O notebook termina salvando `processed_data/`. Sem ele, os notebooks 2–5 falham.
- Se perguntarem "o que do toolkit **não** entrou?": `standard`/`minmax`/`robust` (o `Normalize` só tem normas L1/L2/máx.), a subamostragem por `SAMPLING_RATE`, eventos com sensor inteiramente ausente (o `ImputeMissing` dá erro; o notebook captura com `try/except` e pula o evento) e `DataPersistence` (o toolkit só salva modelos). Veja a tabela do [README](README.md#-relationship-with-the-3w-toolkit).

**Exercícios (Ap. E):** comparar os quatro escalonadores; mudar `SAMPLING_RATE`; implementar a política de procedência do livro.

---

## 7. Notebook 2 — `2_visualization_techniques` (Aula 1, 9 min)

**Na aula (9 min).** *22–31:* 3 min na seção *Toolkit first: a feature view of the windows* (`StatisticalFeatures` × janela achatada, ao vivo); 4 min nas quatro configurações de t-SNE (saída gravada, mudando a perplexidade); 2 min na regra "a projeção é uma escolha, não uma medida". **Pule:** a visualização das janelas por classe (só a saída) e o UMAP (só cite que existe e que precisa do extra `resources`).

**Objetivo:** olhar as janelas em 2D com t-SNE e UMAP — e aprender a desconfiar do que se vê.
**Livro:** Cap. 6, §*Visualização*; Ap. E, §*Caderno 2*. **Toolkit:** `StatisticalFeatures` (seção nova: 30 estatísticas por janela × janela achatada de 900 valores, no mesmo t-SNE). O toolkit não tem t-SNE/UMAP; para gráficos de séries 3W veja o [tutorial de visualização](../introduction_to_3w_toolkit_v2.0.0/6_data_visualization.ipynb).

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

## 8. Notebook 3 — `3_introduction_to_unsupervised_learning` (Aula 2, 15 min)

**Na aula (15 min), só saídas gravadas** (o treino leva 3–6 min). 5 min no OCSVM (diga logo que o toolkit não tem detectores de anomalia e que o `SklearnModelsConfig` recusa o `OneClassSVM`); 7 min no autoencoder (a arquitetura, o erro de reconstrução e onde cai cada limiar); 3 min na AUC do toolkit e no **aviso sobre o nível do sensor**: o AUC quase perfeito não é desempenho do método. **Pule:** o espaço latente, a comparação por dobra e os exercícios.

**Objetivo:** detectar anomalias treinando **só com dados normais** — primeiro com uma abordagem geométrica (OCSVM), depois com um autoencoder LSTM.
**Livro:** Cap. 5 (autoencoders), Cap. 6 (escores de novidade), Cap. 7 (sistema dual), Cap. 12 (Mahalanobis no espaço latente); Ap. E, §*Caderno 3*. **Toolkit:** `metrics.roc_auc_score` calcula a AUC (a curva ROC continua do scikit-learn). O toolkit não tem detectores de anomalia, e o `SklearnModelsConfig` rejeita o `OneClassSVM` (o estimador não tem o método `score` que o protocolo do toolkit exige). O autoencoder é escrito em **PyTorch** (o mesmo framework do toolkit); para `MLP`, `ModelTrainer` e `ModelAssessment` veja o [tutorial 7](../introduction_to_3w_toolkit_v2.0.0/7_model_training_and_evaluation.ipynb).

**Roteiro**

1. **Avaliação por dobra.** Treina-se com os normais de cada dobra; testa-se com normais e anomalias (classes 3, 4 e 8) da mesma dobra. *Nomenclatura:* treinar só com normais e detectar o que foge é **detecção de novidade** (semissupervisionado de uma classe), não agrupamento, que é não supervisionado de fato.
2. **OCSVM.** A janela é achatada num vetor de 900 posições, padronizada e passada ao OCSVM. *Gancho com o livro (Cap. 8):* a mesma família de método obteve **0,65** de acurácia contra **0,87** do *ensemble* binário — e a razão aparece ao olhar a dimensão do vetor: em 900 dimensões as distâncias se concentram. (O RBF não "entende" série temporal: aqui a janela achatada é só um vetor de 900 valores.)
3. **Autoencoder LSTM** (`StableLSTMAutoencoder`). Cada escolha de estabilidade existe porque a alternativa falha (Ap. E): ativação `tanh`; inicialização ortogonal nos pesos recorrentes; corte de gradiente (`clipnorm = 1,0`); taxa de aprendizado `0,0005`; saída sigmoide (casa com o Min-Max); sementes fixas. *Uma perda que vira `NaN` no meio do treino costuma ser culpada nos dados — é sintoma dessas escolhas.* O sistema do Cap. 7, em operação, usa ReLU e funciona por ter um gargalo bem mais estreito (4 dimensões): **as escolhas interagem**.
4. **O limiar.** `AnomalyDetector` implementa dois limiares **diferentes**: o **percentil 95** dos erros normais e **μ + 2σ**. Não são a mesma coisa: numa normal o percentil 95 fica em μ + 1,645σ e μ + 2σ cobre ≈ 97,7%, e o erro de reconstrução nem é normal (é assimétrico), por isso os dois limiares diferem. *Mensagem:* o limiar move o compromisso detecção × alarme falso mais que a arquitetura. **O limiar sai dos normais de treino; os normais de teste servem para medir se ele generaliza.** Por isso a acurácia nos normais de teste (≈ 0,94) fica abaixo de 0,95, e vale mostrar que o limiar calibrado no treino não se transfere perfeitamente. O limiar "ótimo" da curva ROC é o do **índice de Youden** (maximiza TPR − FPR): diga o nome. Os rótulos "dobra com a maior acurácia" e "maior acurácia média" escolhem pelo teste: servem para ilustrar, mas o número é otimista e não é o desempenho esperado. Mostre a curva ROC e onde cada limiar cai.

**Perguntas para a plateia**

- "Se o erro de reconstrução de uma anomalia é alto, o que eu sei? E se for baixo?"
- "O que acontece com o limiar se eu treinar com normais de poços diferentes dos de teste?"

**Pontos de atenção**

- É o notebook mais demorado (≈ 3–6 min na CPU). Mostre as saídas prontas e rode apenas a célula de arquitetura/treino de uma dobra, se quiser demonstrar.
- Com sementes fixas os números se repetem entre execuções na mesma máquina e versões de pacotes; entre máquinas podem variar um pouco (ponto flutuante).
- Não compare os números daqui com os do livro: janelas, sensores, latente e partição são diferentes (tabela no README).
- **Tenha a lista das classes à mão:** 0 normal; 1 aumento abrupto de BSW; 2 fechamento espúrio da DHSV; 3 golfada severa; 4 instabilidade de fluxo; 5 perda rápida de produtividade; 6 restrição rápida no PCK; 7 incrustação no PCK; 8 hidrato na linha de produção; 9 hidrato na linha de serviço.
- No espaço latente (t-SNE), quem fica **fora da distribuição** dos normais são as **anomalias**, não os normais.
- **Cuidado com a AUC.** Com o `Normalize` do toolkit (níveis absolutos mantidos) o autoencoder separa quase perfeitamente (AUC ≈ 1,0; acurácia nas anomalias 1,0). Com o Min-Max por evento do livro, que apaga os níveis, a mesma execução dá AUC 0,96 no autoencoder e 0,97 no OCSVM (0,99 com o `Normalize`). **Não diga "muito bom" para a AUC de 0,99:** ela não é desempenho do método. A diferença mede quanto da detecção vem do nível dos sensores, e não do padrão temporal. A célula de aviso do notebook explica como reproduzir. Parte desse nível é também uma impressão digital do poço (veja o Notebook 4, passo 7).

**Exercícios (Ap. E):** `SAMPLING_RATE` 5 → 1; percentil 95 × μ + 2σ no mesmo erro; Mahalanobis na penúltima camada de um classificador do Notebook 4.

---

## 9. Notebook 4 — `4_introduction_to_supervised_learning` (Aula 2, 20 min)

**Na aula (20 min).** *8 min:* a seção *Toolkit first: one model, end to end* (um modelo, as métricas e a matriz de confusão, tudo do toolkit; roda ao vivo). *6 min:* as tabelas por dobra, com `Test Acc` e `Test Bal. Acc` lado a lado, e a variabilidade entre dobras (só saídas). *6 min:* a seção *Does the sensor level identify the well?* (passo 7 do roteiro; roda em cerca de 15 s, mas use a saída gravada). **Pule:** a análise detalhada de SVM e de redes neurais (cite só a tabela final) e a explicação do laço de treino (um laço único treina os sete modelos em cada dobra). **Antes de comparar os sete modelos** (e não no fim), diga que os números são otimistas porque as dobras são por evento. Compare `Test Acc` com `Test Bal. Acc` (Mensagem 2).

**Objetivo:** treinar e comparar seis modelos clássicos e aprender a avaliá-los.
**Livro:** Cap. 4 (matriz de confusão, acurácia balanceada, validação) e Cap. 8; Ap. E, §*Caderno 4*. **Toolkit:** o notebook abre com **um modelo de ponta a ponta só com o toolkit** (`SklearnModelsConfig`, `ThreeWToolkit.metrics` — acurácia, acurácia balanceada, precisão, revocação e F1 — e `AssessmentVisualization` para a matriz de confusão); depois repete isso por dobra e por algoritmo com o nosso laço (balanceamento do treino, tabelas e painéis de sobreajuste e consistência, que o toolkit não tem). Para `SklearnModels`, `MLP`, `ModelTrainer`, `ModelAssessment` e `Pipeline`, veja os tutoriais [7](../introduction_to_3w_toolkit_v2.0.0/7_model_training_and_evaluation.ipynb) e [8](../introduction_to_3w_toolkit_v2.0.0/8_pipeline_integration.ipynb).

**Roteiro**

1. Classes selecionadas por padrão: **[3, 4, 8]** — golfadas severas, instabilidade de fluxo e hidrato na linha de produção (a classe 0, normal, é o treino do Notebook 3).
2. Modelos: árvore de decisão, floresta aleatória, SVM linear e RBF, e três redes densas (rasa, profunda, regularizada).
3. **Balancear o treino e não o teste.** A assimetria é deliberada: balancear o teste *muda a pergunta* (mede-se sob uma distribuição que não existe na operação). Para o desbalanceamento na avaliação, use acurácia balanceada.
4. **Acurácia × acurácia balanceada.** As tabelas por dobra mostram `Test Acc` e `Test Bal. Acc` lado a lado. *Cap. 4:* um detector inútil atinge 99,7% de acurácia.
5. **Variabilidade entre dobras.** Aponte para a tabela final: a diferença **entre dobras** costuma ser maior que a diferença **entre algoritmos**. Gancho: a saída *share of the test windows* mostra a fração das janelas de teste de cada dobra que pertence às classes 3, 4 e 8 (14,1%, 20,3% e 39,4% nesta versão): as dobras são bem desiguais. É o argumento do livro a favor de avaliar poço a poço e reportar a tabela completa, não só a média.
6. **A classe 4** aparece entre as selecionadas: é a classe sintomática do fundo de todas as tabelas do livro (argumento de taxonomia do Cap. 16).
7. **O nível do sensor é uma impressão digital do poço** (seção *Does the sensor level identify the well?*; ≈ 5 min, a parte mais forte da Aula 2). Roteiro em quatro passos:
   - **A ideia (30 s).** *"O `Normalize` do toolkit usa média e desvio globais, então mantém o nível absoluto de cada sensor. Cada poço tem o seu nível: profundidade, completação, instrumentação. Esse nível funciona como uma impressão digital do poço."*
   - **Teste 1: a janela entrega o poço? (1 min).** Uma floresta tenta adivinhar **de qual poço** vem uma janela (não a falha). São 12 poços, então no chute o acerto é 8%. Com o `Normalize`: ≈ 96%. Com o Min-Max por evento, que apaga o nível: ≈ 57% (sobra a forma da série).
   - **Teste 2: o que isso faz com a avaliação (1,5 min).** *Dobras por evento:* eventos do mesmo poço caem no treino e no teste, e o modelo já "conhece" o poço do teste. *Dobras por poço* (`GroupKFold` com `groups=wells`): o poço inteiro vai para o teste, que é a situação real (um poço novo na operação). Acurácia balanceada:

     | Escalonamento | Dobras por evento | Dobras por poço |
     |---|---|---|
     | `Normalize` (toolkit) | 0,90 | 0,30 |
     | Min-Max por evento | 0,57 | 0,23 |
     | Sem o nível de cada evento, mantendo a escala global | 0,69 | 0,47 |

     Frase-chave: *"De 0,90 para 0,30. A maior parte do acerto vinha de reconhecer o poço, não a falha."*
   - **A ressalva e a lição (1 min).** A amostra é pequena: 113 eventos reais de 12 poços, com desvios grandes por poço (± 0,21), um modelo e uma semente. É um indício, não um ranking de escalonadores. *Cap. 4:* a divisão deve respeitar a unidade que o modelo encontrará na operação; por isso os números altos do Notebook 4 e a AUC ≈ 1 do Notebook 3 não valem para um poço novo. Analogia: *"É como corrigir prova reconhecendo a letra do aluno: funciona na turma que você conhece e falha na turma nova."*
   - **Números de apoio** (tabela de eventos da seção): classe 0 = 37 eventos de 7 poços; classe 3 = 32 eventos de **2** poços; classe 4 = 36 eventos de 5 poços; classe 8 = **8** eventos de 5 poços.

**Pergunta para a plateia:** *"Qual modelo é o melhor?"* — e só depois mostrar a tabela com desvio e consistência.

**Pontos de atenção**

- **Os números do Notebook 4 são otimistas:** as dobras são por evento, então o mesmo poço aparece no treino e no teste (passo 7 do roteiro). Diga isso antes de comparar algoritmos.
- As árvores e a floresta chegam a ≈ 0,99 no treino e a 0,91–0,98 no teste; a lacuna é maior na dobra 3 (0,09 na árvore). Com o `Normalize` do toolkit os níveis absolutos dos sensores são mantidos e as médias por dobra sobem (árvore 0,95, floresta 0,96, SVM RBF 0,97, redes 0,97–0,98) em relação ao Min-Max por evento (0,73–0,86); por isso a variação entre dobras é pequena e o contraste dramático de antes some. Ainda é um bom momento para falar de generalização entre poços, apontando a dobra 3.
- Notebook ≈ 2–3 min na CPU.

**Exercícios (Ap. E):** política de procedência do livro; `selected_classes = None` e comparação das classes 4 e 5.

---

## 10. Notebook 5 — `5_clustering_methods` (Aula 2, 16 min)

**Na aula (16 min), em quatro blocos** (o notebook muda de pergunta no meio, e é aí que a plateia se perde):

- **Bloco A — Toolkit: DTW + hierárquico em eventos (5 min, ao vivo).** Até 15 eventos reais por classe (3, 4 e 8), só o sensor `P-TPT`. Pergunta: *"quais eventos se parecem?"* O objetivo no toolkit é selecionar instâncias bem comportadas, não recuperar as classes: o Rand ajustado baixo é esperado, não uma falha. Só um sensor porque o DTW é caro e aqui é aplicado a séries univariadas; o dendrograma é só a forma de desenhar a árvore.
- **Bloco B — muda a pergunta e os dados (1 min).** Agora cada evento inteiro vira um ponto: a série completa redimensionada para 500 pontos × 3 sensores = 1.500 dimensões (**não são janelas**). Dados: dobra 1, treino e teste juntos, **classes 3, 4 e 8** ("completo" quer dizer série completa, não todas as classes): 150 eventos, que incluem simulados e desenhados. Diga a frase: *"aqui entram simulados e desenhados, ao contrário dos notebooks 2 a 4"* (não há vazamento, porque o notebook é não supervisionado).
- **Bloco C — K-means, MeanShift e DBSCAN (8 + 3 min).** O K escolhido pela silhueta não é 3 (K = 8 nesta versão). O MeanShift acha 11–12 grupos; o DBSCAN marca pontos como ruído, e quanto vira ruído depende muito do raio. Não entre em cada número: o padrão é *grupos compactos, mas que não são as classes*.
- **Bloco D — a tabela-resumo (2 min), o momento-chave.** Põe os métodos lado a lado com três referências. A linha que mais ensina é *the true classes*: as classes verdadeiras têm silhueta ≈ 0,43, **menor** que a dos grupos do K-means (≈ 0,74). A silhueta mede compacidade, não significado.

**Pule:** o K-means em PCA 50D, a seção *Advanced Clustering* e o algoritmo húngaro (no ensaio o notebook ficou incompleto aos 49 min).

**Objetivo:** descobrir estrutura sem rótulos e medir se ela faz sentido.
**Livro:** Cap. 6 (§*Agrupamento* e §*Avaliar agrupamentos sem rótulos*) e Cap. 9 (ciclo de mundo aberto); Ap. E, §*Caderno 5*. **Toolkit:** o notebook **abre** com o sub-pacote `ThreeWToolkit.clustering`, que faz agrupamento **por instância** (qualidade, DTW, hierárquico, dendrograma) em 15 eventos reais por classe; um índice de Rand ajustado baixo contra as classes é esperado, porque o objetivo ali é selecionar instâncias bem comportadas. Só depois vêm K-means, MeanShift e DBSCAN. Veja as demos [clustering_pipeline_examples](../../toolkit/demos/06_clustering_pipeline_examples.ipynb) e [clustering_real_vs_simulated_examples](../../toolkit/demos/07_clustering_real_vs_simulated_examples.ipynb).

**Roteiro**

1. **Dados completos redimensionados para 500 pontos** (diferente dos notebooks anteriores, que usam janelas), PCA para visualização.
2. **K-means:** cotovelo e silhueta para escolher K. *Cap. 9:* em mundo aberto, **K é exatamente o que não se conhece** — o livro treina uma floresta aleatória para *prever* K a partir de oito características do próprio conjunto. Aqui você verá o outro lado: o K sugerido pelos índices raramente coincide com o número de classes (K = 8 com o `Normalize`, K = 4 com o Min-Max por evento, para 3 classes).
3. **MeanShift** (K emerge dos modos; exige a largura de banda; nas 1.500 dimensões do espaço normalizado cada ponto é uma semente) e **DBSCAN** (único que rotula ruído; em alta dimensão o ruído depende muito do raio: com o menor raio da grade, 23% a 40% das séries viram ruído, e com o raio escolhido pela silhueta só 6%; o livro só o usa como comparação).
4. **Avaliação:** silhueta (interna) e índice de Rand ajustado e acurácia com atribuição húngara (externas).
5. **Mensagem do Ap. E:** este notebook é o ciclo de mundo aberto **sem a etapa de rejeição**; você encontra, em minutos, os dois achados do Cap. 9 — uma silhueta alta não garante que os grupos sejam as classes (K-means chega a silhueta 0,74 com o `Normalize`, mas o índice de Rand ajustado é só 0,48; com o Min-Max por evento a silhueta é 0,37 e o índice 0,73) e o número de grupos sugerido não coincide com o de classes. O Cap. 12 explica o porquê; **rodar antes de ler torna a explicação mais convincente**.

**Pergunta para a plateia:** *"Se as classes reais têm silhueta pior que a do K-means, o que a silhueta está medindo?"* (a silhueta do DBSCAN é 0,83 e a do K-means 0,74, mas o Rand ajustado dos dois é cerca de 0,5)

**Pontos de atenção**

- Rápido (< 1 min). As mensagens de concordância fraca ou parcial com a verdade (Rand ajustado ≈ 0,5) são **esperadas** — são o objetivo didático.

**Exercício (Ap. E):** trocar o espaço de entrada pelo **espaço latente do autoencoder** do Notebook 3 e recalcular silhueta e Rand ajustado.

---

## 11. Fechamento das duas aulas

### Aula 1 (4 min)

1. **Mensagem 1:** o número que você reporta depende de decisões anteriores ao modelo. Exemplos desta aula: a procedência dos eventos, o escalonamento (o `Normalize` do toolkit mantém o nível do sensor), as dobras por evento e o janelamento.
2. **Primeira metade da mensagem 3:** a projeção t-SNE/UMAP é uma escolha, não uma medida (Notebook 2).
3. **Gancho:** "agora vamos treinar um modelo e ver o que o número esconde" (Aula 2).
4. **Exercícios para casa (Ap. E):** 1 (comparar os quatro escalonadores) e 2 (`SAMPLING_RATE`).

### Aula 2 (6 min)

1. **Mensagem 2** (Notebook 4): acurácia simples engana e a média esconde variabilidade; o nível do sensor identifica o poço, então a divisão deve respeitar a unidade que o modelo encontrará na operação.
2. **Mensagem 3:** visualizar, agrupar e detectar novidade são ferramentas de inspeção, não de medida. Retome a projeção como escolha (Notebook 2, da Aula 1) e dê um exemplo de cada notebook de hoje: o limiar e o AUC do nível do sensor (3), a silhueta alta sem coincidir com as classes e o K diferente do número de classes (5).
3. **A ponte para o livro:** Parte I (fundamentos) = o que fizemos nas duas aulas; Parte II (Caps. 7–9), o sistema dual, o *ensemble* binário e o aprendizado de mundo aberto; Parte III, a classificação em mundo fechado, a segmentação com U-Net e Mahalanobis no espaço latente; Parte IV, a explicabilidade por modelos de linguagem; Parte V, a síntese e os requisitos de operação.
4. **Próximos passos:** (a) o tutorial [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md) (escrito para o toolkit 2.0.0) e as demos do toolkit 3.x para construir *pipelines* completos; (b) os exercícios guiados do Ap. E (todos os cadernos terminam com os deles), em especial o 3 e o 5 (Notebook 4) e o 6 (Notebook 5); (c) citar o livro e o 3W.
5. **Convite:** contribuições e dúvidas pelo repositório do projeto ([petrobras/3W](https://github.com/petrobras/3W)).

---

## 12. Plano B (se algo falhar)

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| `ModuleNotFoundError: ThreeWToolkit` | Ambiente sem o toolkit instalado | `uv pip install -e "../../.[notebooks,resources]" jupyter` no ambiente do kernel; reiniciar o kernel |
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
- **"Posso usar isto com outro conjunto de dados?"** Sim, a parametrização está em `src/config.py`; o pipeline é configurável (sensores, janela, escalonamento). Para conjuntos fora do 3W, troque a célula de carga do Notebook 1 (o `ParquetDataset`).
- **"O toolkit já faz tudo isto?"** Parte: carga, métricas, visualização, *pipelines* e agrupamento por instância. O recurso mantém localmente o que o toolkit não oferece ou faria de outra forma (README, seção *Relationship with the 3W Toolkit*).
- **"Onde está o código do autoencoder do livro?"** Este recurso traz uma versão didática em PyTorch; os experimentos do livro estão descritos no Ap. D.

## 14. Mudanças em relação à versão anterior (`introduction_to_ml_applied_to_mts`)

Resumo, para quem já conhecia a versão anterior deste recurso (a pasta [`../introduction_to_ml_applied_to_mts`](../introduction_to_ml_applied_to_mts/README.md), que continua como estava), do que passou a vir do [3W Toolkit](../../toolkit/ThreeWToolkit/README.md) (agora o pacote oficial) e do que continua local. Regra seguida: **só se substituiu o que não altera os resultados**; o que mudaria foi mantido (e explicado no [README](README.md#-relationship-with-the-3w-toolkit)).

### Acrescentado a partir do toolkit

| Componente do toolkit | Onde entra | O que muda para quem apresenta |
|---|---|---|
| `dataset.ParquetDataset` / `ParquetDatasetConfig` | Notebook 1 (exploração, carga e dobras) | A descoberta e a leitura dos arquivos passam pelo toolkit, que também filtra por procedência (**real**, **simulada**, **desenhada**). O Notebook 1 abre mostrando o `ParquetDataset` diretamente. Se o dataset 2.0.0 faltar em `../../dataset/`, o toolkit o baixa do Figshare. |
| `data_visualization.PlotMultipleSeries`, `CorrelationHeatmap` | Notebook 1 | Substituem os gráficos locais removidos. `ThreeWChart` (interativo) é citado, mas não gravado nas saídas. |
| `preprocessing.Normalize` | Notebook 1, `DEFAULT_SCALING_METHOD = "normalizer"` e `NORMALIZE_NORM = "l2"` | **Novo escalonador padrão.** Ajustado só nos eventos de treino de cada dobra e aplicado aos de teste (sem vazamento). Os escalonadores `standard`, `minmax` e `robust` continuam disponíveis e são ajustados por evento. O autoencoder (saída sigmoide) recebe ainda um reescalonamento Min-Max das janelas normais no Notebook 3. |
| `preprocessing.ImputeMissing` | Notebook 1 | Preenchimento *ffill* por evento antes da normalização; eventos com um sensor inteiramente ausente são pulados (`try/except`). |
| `utils.KFoldSplitter` | Notebook 1 | As dobras são feitas **sobre os eventos reais** (estratificadas por classe, sem vazamento entre janelas); a política "treina com real + simulado, testa só com real" é escrita no próprio notebook, sobre o divisor do toolkit (a classe `CrossValidator` foi removida). |
| `feature_extraction.Windowing` | Notebook 1 | As janelas completas saem do toolkit (`pad_start=False`, `overlap = WINDOW_OVERLAP`); `windows_to_frames` (`src/events.py`) as reempacota em um DataFrame por janela, na mesma contagem de antes. O Notebook 1 mostra o formato nativo do toolkit. |
| `feature_extraction.StatisticalFeatures` | Notebook 2 | Seção nova: t-SNE das 30 estatísticas por janela × janela achatada (900 valores). |
| `models.SklearnModelsConfig` | Notebook 4 | Os sete classificadores do scikit-learn são criados pela configuração do toolkit, num dicionário `MODELS` e um laço por dobra (o `OneClassSVM` é recusado pelo validador do toolkit e usado direto do scikit-learn no Notebook 3, ver abaixo). O Notebook 4 abre com um modelo de ponta a ponta. |
| `metrics.accuracy_score`, `balanced_accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` | Notebooks 3 e 4 | Mesmos valores de antes para acurácia; as tabelas por dobra do Notebook 4 têm a coluna `Test Bal. Acc` (Cap. 4); a AUC do Notebook 3 sai do toolkit (a curva continua do scikit-learn). |
| `assessment.AssessmentVisualization` | Notebook 4 | Matriz de confusão normalizada do modelo de ponta a ponta. |
| `clustering` (`InstanceQualityFilter`, `TimeSeriesResampler`, `TimeSeriesScaler`, `DistanceComputer`, `HierarchicalClusterer`) e `DendrogramPlot` | Notebook 5 | O notebook abre com o agrupamento hierárquico por DTW de eventos reais (`P-TPT`) e só depois apresenta K-means/MeanShift/DBSCAN. |
| `pyproject.toml` do repositório | — | O recurso usa o toolkit instalado (`uv pip install -e "../../.[notebooks,resources]" jupyter`). O extra `resources` instala o UMAP e o `notebooks` o kernel e o `nbconvert`. **TensorFlow não é mais necessário.** |

Efeitos colaterais de usar o toolkit: ao ser importado, ele configura o log em nível INFO; por isso `src/__init__.py` rebaixa o log raiz para WARNING, para as saídas dos notebooks ficarem legíveis.

### Removido do `src/` (o toolkit já cobre)

Os notebooks chamam o toolkit direto, então os auxiliares que o embrulhavam saíram do `src/` (cerca de 7.000 das 12.400 linhas), junto com código que ninguém chamava:

| Removido | Use no lugar |
|---|---|
| `DataVisualizer.plot_correlation_matrix` | `data_visualization.CorrelationHeatmap` |
| `DataVisualizer.plot_time_series_comparison` | `data_visualization.PlotMultipleSeries` |
| `DataPreprocessor`, `CrossValidator`, `DataLoader` (e os gráficos do `DataVisualizer`) | `Normalize`, `ImputeMissing`, `Windowing`, `KFoldSplitter`, `ParquetDataset`, `PlotMultipleSeries`, chamados direto no Notebook 1 |
| `SupervisedClassifier`, `enhanced_fold_analysis` e as análises por família do Notebook 4, e `build_sklearn_model` | `SklearnModelsConfig` e as métricas do toolkit, num laço único no Notebook 4 |
| `DistanceAnomalyDetector` | `Pipeline` do scikit-learn com `OneClassSVM`, no Notebook 3 (o toolkit recusa o `OneClassSVM`) |

### Avaliado e mantido local (e por quê)

| Local | Equivalente no toolkit | Motivo |
|---|---|---|
| Escalonadores `standard`, `minmax`, `robust` | — | Não existem no toolkit (só L1/L2/máx. em `Normalize`). Ficam como alternativas ao `Normalize`, e a comparação dos quatro métodos é o exercício 1 do Ap. E. |
| Limpeza e subamostragem de cada evento (`src/events.py`) | `preprocessing.ImputeMissing` (só preenche) | O toolkit não descarta linhas sem rótulo nem mantém uma linha a cada `SAMPLING_RATE`. |
| Eventos com sensor inteiramente ausente | `preprocessing.ImputeMissing` | O toolkit dá erro; o recurso descarta o evento e segue. |
| Redimensionamento para 500 pontos (K-means, DBSCAN) | `clustering.TimeSeriesResampler` | Método diferente (interpolação para tamanho fixo × fatiar por passo). |
| t-SNE/UMAP, K-means/MeanShift/DBSCAN (índices de silhueta e Rand, Hungarian), aumento de dados | — | O toolkit não oferece. |
| OCSVM e limiares do autoencoder, curva ROC | — | O toolkit não tem detectores de anomalia (só `roc_auc_score`). O OCSVM é o `OneClassSVM` do scikit-learn usado direto no Notebook 3: o `SklearnModelsConfig` o rejeita porque o estimador não tem o método `score` exigido pelo protocolo do toolkit. |
| Autoencoder LSTM | `models.MLP`, `trainer.TorchTrainer` | O toolkit não tem autoencoder recorrente. O modelo é escrito em **PyTorch**, o framework do toolkit. |
| Balanceamento do treino e achatamento das janelas do Notebook 4 (`src/supervised_classification.py`), tabelas e painéis de sobreajuste e consistência | `trainer.SklearnTrainer`, `assessment.ModelAssessment` | Os *trainers* do toolkit trabalham com eventos e não com janelas achatadas; os painéis comparativos entre modelos não existem no toolkit. |
| `DataPersistence`, `save_complete_pipeline` | — | O toolkit só salva modelos (`ModelRecorder`). |

### Outras mudanças que afetam a apresentação

- **Autoencoder em PyTorch** (antes TensorFlow/Keras): mesma API e mesmas decisões de estabilidade; os números do Notebook 3 mudaram em relação às gravações do workshop.
- **Reprodutibilidade:** a amostragem de janelas do Notebook 3 agora usa semente fixa; execuções repetidas dão os mesmos números.
- **Escalonamento depois da divisão:** o escalonamento passou a ser feito *depois* de montar as dobras (Notebook 1, passo 5c), com as estatísticas do treino de cada dobra, como recomenda o livro.
- **Fronteira real/simulado exata:** o Notebook 1 monta as dobras direto sobre os eventos reais que sobreviveram à limpeza; antes (no `CrossValidator`, removido) a fronteira era estimada por proporção e podia colocar eventos simulados no teste "só real".
- **Correções:** `boxplot(labels=...)` (quebrava com o matplotlib atual), gráfico do OCSVM (acurácias complementares e "ROC invertido"), chamadas de gráfico de K-means com assinatura antiga (avisos no Notebook 5) e instalação de pacotes via `subprocess` no Notebook 2.
- **Conteúdo:** cada notebook ganhou referências ao livro (capítulos e seções) e termina com os exercícios do Ap. E; o README ganhou a citação do livro e a tabela de diferenças em relação aos experimentos dele.

---

## 15. Referências

- **Livro:** Lopes, L. G. O. et al. (2026). *Aprendizado de Mundo Aberto em Poços de Petróleo.* DOI [10.5281/zenodo.22306796](https://doi.org/10.5281/zenodo.22306796). Cap. 3 (dados), 4 (avaliação), 5 (aprendizado profundo), 6 (novidade e agrupamento), 7 (sistema dual), 8 (ensemble binário), 9 (mundo aberto), 12 (Mahalanobis), 16 (síntese); Ap. D (reprodução) e Ap. E (prática).
- **3W Project:** [github.com/petrobras/3W](https://github.com/petrobras/3W); [3W Dataset](../../dataset/README.md); [3W Toolkit](../../toolkit/ThreeWToolkit/README.md); [3W Dataset no Figshare](https://figshare.com/projects/3W_Dataset/251195).
- **Tutoriais do toolkit:** [Introduction to 3WToolkit v2.0.0](../introduction_to_3w_toolkit_v2.0.0/README.md) (escritos para o toolkit 2.0.0) e as [demos do toolkit 3.x](../../toolkit/demos/).
- **Este recurso:** [README](README.md).
