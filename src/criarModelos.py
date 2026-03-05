# --------------------------------------------
# Passo 3: Criar modelos de previsão
# --------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def criar_modelos():
    # Criar modelos vazios (a serem treinados)
    #passo 2: criar a IA

    modelo_arvoredecisao = RandomForestClassifier(random_state=42)
    modelo_knn = KNeighborsClassifier()
    return modelo_arvoredecisao, modelo_knn
