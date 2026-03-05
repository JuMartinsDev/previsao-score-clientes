# --------------------------------------------
# Passo 6: Fazer previsões com novos clientes
# --------------------------------------------
import pandas as pd

def prever_novos_clientes(caminhoNovosClientes, codificador_profissao, codificador_mix, codificador_comportamento, modelo, codificador_score):
    """
    caminhoNovosClientes: caminho para CSV dos novos clientes
    codificador_*: LabelEncoders usados para treinar o modelo
    modelo: melhor modelo treinado
    codificador_score: LabelEncoder do score de crédito
    """
    # Configurar pandas para exibir mais colunas e linhas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 20)

    # Importar novos clientes
    novos_clientes = pd.read_csv(caminhoNovosClientes)
    print("\nNovos clientes (5 primeiros registros):")
    print(novos_clientes.head())

    # Aplicar mesmo codificador usado no treino
    novos_clientes["profissao"] = codificador_profissao.transform(novos_clientes["profissao"])
    novos_clientes["mix_credito"] = codificador_mix.transform(novos_clientes["mix_credito"])
    novos_clientes["comportamento_pagamento"] = codificador_comportamento.transform(novos_clientes["comportamento_pagamento"])

    # Preparar dados de entrada (remover coluna id_cliente, se existir)
    x_novos = novos_clientes.drop(columns=["id_cliente"], errors='ignore')

    # Fazer previsões
    previsao = modelo.predict(x_novos)

    # Decodificar previsões para labels legíveis
    previsao_labels = codificador_score.inverse_transform(previsao)

    # Adicionar coluna com previsão ao DataFrame
    novos_clientes["previsao_score_credito"] = previsao_labels

    # Mostrar 10 primeiros registros com ID e previsão, incluindo colunas importantes
    colunas_mostrar = ["id_cliente", "profissao", "mix_credito", "comportamento_pagamento", "previsao_score_credito"]
    colunas_existentes = [col for col in colunas_mostrar if col in novos_clientes.columns]

    print("\nPrevisão do score de crédito para novos clientes (10 primeiros registros):")
    print(novos_clientes[colunas_existentes].head(10))

    return novos_clientes