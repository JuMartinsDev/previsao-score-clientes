# --------------------------------------------
# Passo 4: Treinar os modelos
# --------------------------------------------
def treinar_modelos(modelos, x_treino, y_treino):
    modelo_arvoredecisao, modelo_knn = modelos
    # Treinar Random Forest
    modelo_arvoredecisao.fit(x_treino, y_treino)
    # Treinar KNN
    modelo_knn.fit(x_treino, y_treino)
    return modelo_arvoredecisao, modelo_knn