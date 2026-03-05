# 📊 Projeto: Previsão de Score de Clientes - Machine Learning

## Descrição do Projeto
Este projeto tem como objetivo analisar dados de clientes de um banco e prever o **score de crédito** de novos clientes utilizando técnicas de **Machine Learning**. O score define se um cliente é **bom, padrão ou ruim**, auxiliando o banco a conceder crédito de forma segura.

O pipeline do projeto inclui:  
1. Importação da base de clientes históricos.  
2. Tratamento e codificação de dados categóricos.  
3. Treinamento de modelos de classificação (`Random Forest` e `KNN`).  
4. Avaliação de modelos e escolha do melhor.  
5. Previsão do score de crédito de novos clientes.  
6. Exibição de resultados de forma legível e interpretável.

> o objetivo é obter previsões confiáveis e interpretáveis.

---

## Estrutura do Projeto

```

previsaoML/
│
├─ base/
│  ├─ clientes.csv           # Dados históricos dos clientes
│  └─ novos_clientes.csv     # Dados dos clientes a prever
│
├─ importarBase.py           # Função para carregar o CSV de clientes
├─ tratarBase.py             # Tratamento da base e codificação de dados
├─ criarModelos.py           # Criação dos modelos de ML
├─ treinarModelos.py         # Treinamento dos modelos
├─ avaliarModelos.py         # Avaliação de modelos e escolha do melhor
├─ preverModelos.py          # Função para prever score de novos clientes
├─ pipeline.py               # Pipeline completo integrando todas as etapas
└─ README.md                 # Documentação do projeto

````

---

## Tecnologias e Bibliotecas

- Python 3.x  
- Pandas  
- Scikit-learn (`RandomForestClassifier`, `KNeighborsClassifier`, `LabelEncoder`, `train_test_split`, `metrics`)  

---

## Como Rodar o Projeto

1. Clone o repositório:  
```bash
git clone <URL_DO_REPOSITORIO>
cd previsaoML
````

2. Instale as dependências:

```bash
pip install pandas scikit-learn
```

3. Certifique-se de que os arquivos `clientes.csv` e `novos_clientes.csv` estão na pasta `base/`.

4. Execute o pipeline principal:

```bash
python pipeline.py
```

---

## Funcionalidades

### 1️⃣ Importar Base

* Carrega os dados históricos (`clientes.csv`) e exibe os 5 primeiros registros.

### 2️⃣ Tratamento de Dados

* Codifica colunas categóricas: `profissao`, `mix_credito`, `comportamento_pagamento`, `score_credito`.
* Separa **X** (atributos) e **Y** (score).
* Divide os dados em treino e teste (70% treino, 30% teste).

### 3️⃣ Criação de Modelos

* Cria dois modelos de classificação: `Random Forest` e `KNN`.

### 4️⃣ Treinamento de Modelos

* Treina ambos os modelos com os dados de treino.

### 5️⃣ Avaliação e Seleção do Melhor Modelo

* Compara acurácia dos modelos no conjunto de teste.
* Exibe **acurácia, primeiros 10 resultados, matriz de confusão e relatório de classificação**.
* Escolhe automaticamente o melhor modelo.

### 6️⃣ Previsão de Novos Clientes

* Aplica os codificadores nos dados de novos clientes.
* Faz previsão de score usando o melhor modelo.
* Exibe **10 primeiros clientes com ID, profissão, mix de crédito, comportamento de pagamento e score previsto**.

---

## Exemplo de Saída no Terminal

**Acurácia e avaliação:**

```
Acurácia Random Forest: 0.82
Acurácia KNN: 0.74
Melhor modelo: Random Forest

Primeiros 10 resultados Random Forest:
Real: Good, Previsto: Good
Real: Poor, Previsto: Poor
...
```

**Previsão de novos clientes:**

```
   id_cliente    profissao   mix_credito           comportamento_pagamento previsao_score_credito
0        101  empresario        Normal     baixo_gasto_pagamento_baixo               Poor
1        102    advogado         Ruim      baixo_gasto_pagamento_medio              Poor
2        103  empresario        Normal      baixo_gasto_pagamento_alto           Standard
...
```

---

## Observações

* O pipeline é **modular**, permitindo adicionar novos modelos ou codificadores facilmente.
* A saída é configurada para ser legível, mostrando **labels decodificados** ao invés de números.
* O modelo `Random Forest` geralmente apresenta melhor desempenho, mas o KNN também é avaliado para comparação.

```
