# --------------------------------------------
# Passo 5: Avaliar e escolher o melhor modelo
# --------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def avaliar_modelos(modelos, x_teste, y_teste, codificador_score):
    """
    Avalia os modelos treinados e retorna o melhor modelo baseado na acurácia.
    Exibe resultados decodificados e métricas.
    """
    # Separar os modelos
    modelo_arvoredecisao, modelo_knn = modelos

    # Fazer previsões
    previsao_arvore = modelo_arvoredecisao.predict(x_teste)
    previsao_knn = modelo_knn.predict(x_teste)

    # Avaliar acurácia
    acuracia_arvore = accuracy_score(y_teste, previsao_arvore)
    acuracia_knn = accuracy_score(y_teste, previsao_knn)

    print(f"Acurácia Random Forest: {acuracia_arvore:.2f}")
    print(f"Acurácia KNN: {acuracia_knn:.2f}")

    # Escolher melhor modelo
    if acuracia_arvore >= acuracia_knn:
        melhor_modelo = modelo_arvoredecisao
        print("Melhor modelo: Random Forest")
    else:
        melhor_modelo = modelo_knn
        print("Melhor modelo: KNN")

    # Decodificar previsões para mostrar labels reais
    y_teste_labels = codificador_score.inverse_transform(y_teste)
    previsao_arvore_labels = codificador_score.inverse_transform(previsao_arvore)
    previsao_knn_labels = codificador_score.inverse_transform(previsao_knn)

    # Mostrar primeiros 10 resultados Random Forest
    print("\nPrimeiros 10 resultados Random Forest:")
    for real, pred in zip(y_teste_labels[:10], previsao_arvore_labels[:10]):
        print(f"Real: {real}, Previsto: {pred}")

    # Mostrar matriz de confusão
    print("\nMatriz de Confusão - Random Forest:")
    print(confusion_matrix(y_teste_labels, previsao_arvore_labels))

    # Mostrar relatório completo de classificação
    print("\nRelatório de Classificação - Random Forest:")
    print(classification_report(y_teste_labels, previsao_arvore_labels))

    return melhor_modelo