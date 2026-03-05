# Passo 0: Entender o desafio da empresa
# --------------------------------------------
# Passo 1: Importar a base de dados
# -------------------------------------------
import pandas as pd

caminhoClientes = "./base/clientes.csv"

def base(caminhoClientes):
    df = pd.read_csv(caminhoClientes)
    print("Dados do DataFrame (5 primeiros registros):")
    print(df.head())
    return df