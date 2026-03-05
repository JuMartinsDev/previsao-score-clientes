# --------------------------------------------
# Pipeline completo
# --------------------------------------------
from importarBase import base
from tratarBase import codificadorDados
from criarModelos import criar_modelos
from treinarModelos import treinar_modelos
from avaliarModelos import avaliar_modelos
from preverModelos import prever_novos_clientes

# Caminho do CSV dos clientes
caminhoClientes = "./base/clientes.csv"
# Caminho do CSV dos novos clientes
caminhoNovosClientes = "./base/novos_clientes.csv"

# Passo 1
df = base(caminhoClientes)

# Passo 2
x_treino, x_teste, y_treino, y_teste, codificador_profissao, codificador_mix, codificador_comportamento, codificador_score = codificadorDados(df)

# Passo 3
modelos = criar_modelos()

# Passo 4
modelos_treinados = treinar_modelos(modelos, x_treino, y_treino)

# Passo 5 
melhor_modelo = avaliar_modelos(modelos_treinados, x_teste, y_teste, codificador_score)

# Passo 6
previsoes = prever_novos_clientes(
    caminhoNovosClientes, 
    codificador_profissao, 
    codificador_mix, 
    codificador_comportamento, 
    melhor_modelo,
    codificador_score
)