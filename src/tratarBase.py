# --------------------------------------------
# Passo 2: Tratar e preparar a base de dados
# --------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def codificadorDados(df):
    # Transformar colunas de texto em números usando LabelEncoder

    # Profissão
    codificador_profissao = LabelEncoder()
    df["profissao"] = codificador_profissao.fit_transform(df["profissao"])

    # Mix de crédito
    codificador_mix = LabelEncoder()
    df["mix_credito"] = codificador_mix.fit_transform(df["mix_credito"])

    # Comportamento de pagamento
    codificador_comportamento = LabelEncoder()
    df["comportamento_pagamento"] = codificador_comportamento.fit_transform(df["comportamento_pagamento"])

    # print("Informações do DataFrame codificado:")
    # print(df.info())

# A IA precisa analisar uma base, um hisótico para aprender como ele fara
# A primeira coisa que a IA precisa é separar oq ele irá utilizar e oq iremos prever

    # Codificar score_credito (Good, Ok, Ruim) como números
    codificador_score = LabelEncoder()
    df["score_credito"] = codificador_score.fit_transform(df["score_credito"])

    # Separar X e Y
    # Y = o que você irá prever
    y = df["score_credito"]
    # X = colunas que vai usar para previsão 
    x = df.drop(columns=["score_credito", "id_cliente"])

    # Dividir dados em treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    return x_treino, x_teste, y_treino, y_teste, codificador_profissao, codificador_mix, codificador_comportamento, codificador_score