# Passo 0: Entender o desafio da empresa
# Passo 1: Importar a base de dados

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

caminhoClientes = "./base/clientes.csv"
df = pd.read_csv(caminhoClientes)

# Passo 2: Tratar e preparar a base de dados para inteligência artificial
# quero prever a coluna "score_credito" - todas as outras colunas devem ser números, só a score números
# transformando colunas de texto em número com o LabelEncoder

#profissao
codificador_profissao = LabelEncoder()
df["profissao"] = codificador_profissao.fit_transform(df["profissao"])

#mix_credito
codificador_mix = LabelEncoder()
df["mix_credito"] = codificador_mix.fit_transform(df["mix_credito"])

#comportamento_pagamento
codificador_comportamento = LabelEncoder()
df["comportamento_pagamento"] = codificador_comportamento.fit_transform(df["comportamento_pagamento"])

print(df.info())

# A IA precisa analisar uma base, um hisótico para aprender como ele fara
# A primeira coisa que a IA precisa é separar oq ele irá utilizar e oq iremos prever

# Y = o que você irá prever
y = df["score_credito"]
# X = colunas que vai usar para previsão 
x = df.drop(columns=["score_credito", "id_cliente"])

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)


# Passo 3: Criar modelo de previsão ou modelo de IA -> Ruim, Ok, Boa
#passo 1: importar a IA - import

#passo 2: criar a IA
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
    
#passo 3: treinar a IA
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# Passo 4: Avaliar e escolher o melhor modelo

#previsao
previsao_arvore = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

#comparar a previsão com o y_teste
from sklearn.metrics import accuracy_score

print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))

# Passo 5: Fazer previsões
#melhor modelo = Arvore de decisao

#importar os novos clientes
baseNovosClientes = "./base/novos_clientes.csv"
novos_clientes = pd.read_csv(baseNovosClientes)
print(novos_clientes.head())

#codificador já foi criado
novos_clientes["profissao"] = codificador_profissao.transform(novos_clientes["profissao"])
novos_clientes["mix_credito"] = codificador_mix.transform(novos_clientes["mix_credito"])
novos_clientes["comportamento_pagamento"] = codificador_comportamento.transform(novos_clientes["comportamento_pagamento"])

x_novos = novos_clientes.drop(columns=["id_cliente"])
previsao = modelo_arvoredecisao.predict(x_novos)
print(previsao)

