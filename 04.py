import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suprimir warnings de convergência para MLPClassifier
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Carregar o conjunto de dados Iris
iris = load_iris()

# Definir os percentuais de dados de treinamento
percentuais_treinamento = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Definir o número de repetições
num_repeticoes = 20

# Inicializar dicionários para armazenar os resultados
resultados_knn_1 = {}
resultados_knn_3 = {}
resultados_knn_5 = {}
resultados_dmc = {}
resultados_mlp = {}

# Loop através dos diferentes percentuais de treinamento
for percentual_treinamento in percentuais_treinamento:
    # Inicializar listas para armazenar resultados de cada repetição
    acuracias_knn_1 = []
    acuracias_knn_3 = []
    acuracias_knn_5 = []
    acuracias_dmc = []
    acuracias_mlp = []

    for _ in range(num_repeticoes):
        # Dividir os dados em conjunto de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=1-percentual_treinamento)

        # KNN com k=1
        knn_1 = KNeighborsClassifier(n_neighbors=1)
        acuracia_knn_1 = np.mean(cross_val_score(knn_1, X_train, y_train, cv=5))
        acuracias_knn_1.append(acuracia_knn_1)

        # KNN com k=3
        knn_3 = KNeighborsClassifier(n_neighbors=3)
        acuracia_knn_3 = np.mean(cross_val_score(knn_3, X_train, y_train, cv=5))
        acuracias_knn_3.append(acuracia_knn_3)

        # KNN com k=5
        knn_5 = KNeighborsClassifier(n_neighbors=5)
        acuracia_knn_5 = np.mean(cross_val_score(knn_5, X_train, y_train, cv=5))
        acuracias_knn_5.append(acuracia_knn_5)

        # DMC
        dmc = QuadraticDiscriminantAnalysis()
        acuracia_dmc = np.mean(cross_val_score(dmc, X_train, y_train, cv=5))
        acuracias_dmc.append(acuracia_dmc)

        # MLP
        mlp = MLPClassifier()
        parameters = {'hidden_layer_sizes': [(10,), (50,), (100,)],
                      'activation': ['tanh', 'relu'],
                      'solver': ['sgd', 'adam'],
                      'alpha': [0.0001, 0.05]}
        mlp_grid = GridSearchCV(mlp, parameters, cv=5)
        mlp_grid.fit(X_train, y_train)
        acuracia_mlp = mlp_grid.best_score_
        acuracias_mlp.append(acuracia_mlp)

    # Armazenar as estatísticas de desempenho para cada percentual de treinamento
    resultados_knn_1[percentual_treinamento] = {
        'min': np.min(acuracias_knn_1),
        'max': np.max(acuracias_knn_1),
        'media': np.mean(acuracias_knn_1),
        'desvio_padrao': np.std(acuracias_knn_1)
    }

    resultados_knn_3[percentual_treinamento] = {
        'min': np.min(acuracias_knn_3),
        'max': np.max(acuracias_knn_3),
        'media': np.mean(acuracias_knn_3),
        'desvio_padrao': np.std(acuracias_knn_3)
    }

    resultados_knn_5[percentual_treinamento] = {
        'min': np.min(acuracias_knn_5),
        'max': np.max(acuracias_knn_5),
        'media': np.mean(acuracias_knn_5),
        'desvio_padrao': np.std(acuracias_knn_5)
    }

    resultados_dmc[percentual_treinamento] = {
        'min': np.min(acuracias_dmc),
        'max': np.max(acuracias_dmc),
        'media': np.mean(acuracias_dmc),
        'desvio_padrao': np.std(acuracias_dmc)
    }

    resultados_mlp[percentual_treinamento] = {
        'min': np.min(acuracias_mlp),
        'max': np.max(acuracias_mlp),
        'media': np.mean(acuracias_mlp),
        'desvio_padrao': np.std(acuracias_mlp)
    }

# Imprimir resultados
print("Resultados para KNN (k=1):")
for percentual, estatisticas in resultados_knn_1.items():
    print(f"Percentual de treinamento: {percentual:.0%}")
    print(f"Mínimo: {estatisticas['min']:.2f}")
    print(f"Máximo: {estatisticas['max']:.2f}")
    print(f"Média: {estatisticas['media']:.2f}")
    print(f"Desvio padrão: {estatisticas['desvio_padrao']:.2f}")
    print()

# Repetir para os outros algoritmos
print("Resultados para KNN (k=3):")
for percentual, estatisticas in resultados_knn_3.items():
    print(f"Percentual de treinamento: {percentual:.0%}")
    print(f"Mínimo: {estatisticas['min']:.2f}")
    print(f"Máximo: {estatisticas['max']:.2f}")
    print(f"Média: {estatisticas['media']:.2f}")
    print(f"Desvio padrão: {estatisticas['desvio_padrao']:.2f}")
    print()

print("Resultados para KNN (k=5):")
for percentual, estatisticas in resultados_knn_5.items():
    print(f"Percentual de treinamento: {percentual:.0%}")
    print(f"Mínimo: {estatisticas['min']:.2f}")
    print(f"Máximo: {estatisticas['max']:.2f}")
    print(f"Média: {estatisticas['media']:.2f}")
    print(f"Desvio padrão: {estatisticas['desvio_padrao']:.2f}")
    print()

print("Resultados para DMC:")
for percentual, estatisticas in resultados_dmc.items():
    print(f"Percentual de treinamento: {percentual:.0%}")
    print(f"Mínimo: {estatisticas['min']:.2f}")
    print(f"Máximo: {estatisticas['max']:.2f}")
    print(f"Média: {estatisticas['media']:.2f}")
    print(f"Desvio padrão: {estatisticas['desvio_padrao']:.2f}")
    print()

print("Resultados para MLP:")
for percentual, estatisticas in resultados_mlp.items():
    print(f"Percentual de treinamento: {percentual:.0%}")
    print(f"Mínimo: {estatisticas['min']:.2f}")
    print(f"Máximo: {estatisticas['max']:.2f}")
    print(f"Média: {estatisticas['media']:.2f}")
    print(f"Desvio padrão: {estatisticas['desvio_padrao']:.2f}")
    print()
