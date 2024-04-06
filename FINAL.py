import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
import warnings
import datetime
import time

# Suprimir warnings de convergência para MLPClassifier
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

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

# Inicializar o tempo de início
start_time = time.time()

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
        try:
            dmc = QuadraticDiscriminantAnalysis()
            acuracia_dmc = np.mean(cross_val_score(dmc, X_train, y_train, cv=5))
            acuracias_dmc.append(acuracia_dmc)
        except Exception as e:
            acuracias_dmc.append(np.nan)

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
        'min': np.nanmin(acuracias_dmc),
        'max': np.nanmax(acuracias_dmc),
        'media': np.nanmean(acuracias_dmc),
        'desvio_padrao': np.nanstd(acuracias_dmc)
    }

    resultados_mlp[percentual_treinamento] = {
        'min': np.min(acuracias_mlp),
        'max': np.max(acuracias_mlp),
        'media': np.mean(acuracias_mlp),
        'desvio_padrao': np.std(acuracias_mlp)
    }

# Função para plotar os gráficos de barras
def plot_bar_chart(data, title):
    plt.figure(figsize=(10, 6))
    for percentual, estatisticas in data.items():
        plt.bar(percentual, estatisticas['media'], yerr=estatisticas['desvio_padrao'], capsize=5, label=f"{percentual:.0%}")
    plt.xlabel('Percentual de Treinamento')
    plt.ylabel('Acurácia Média')
    plt.title(title)
    plt.legend(title='Percentual de Treinamento')
    plt.show()

# Função para plotar o gráfico de linhas
def plot_line_chart(data, title):
    plt.figure(figsize=(10, 6))
    for percentual, estatisticas in data.items():
        plt.plot(percentuais_treinamento, [estatisticas['media'] for _ in range(len(percentuais_treinamento))], label=f"{percentual:.0%}")
    plt.xlabel('Percentual de Treinamento')
    plt.ylabel('Acurácia Média')
    plt.title(title)
    plt.legend(title='Percentual de Treinamento')
    plt.show()

# Menu interativo
while True:
    print("Escolha o tipo de gráfico:")
    print("1. Gráfico de Barras para Média de Acurácia (KNN k=1)")
    print("2. Gráfico de Linhas para Média de Acurácia (KNN k=1)")
    print("3. Sair")
    escolha = input("Digite o número correspondente à sua escolha: ")

    if escolha == '1':
        plot_bar_chart(resultados_knn_1, "Gráfico de Barras para Média de Acurácia (KNN k=1)")
    elif escolha == '2':
        plot_line_chart(resultados_knn_1, "Gráfico de Linhas para Média de Acurácia (KNN k=1)")
    elif escolha == '3':
        print("Saindo...")
        break
    else:
        print("Escolha inválida. Por favor, digite o número correspondente à sua escolha.")

# Calcular e imprimir tempo de execução
end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo de execução: {datetime.timedelta(seconds=round(execution_time))}")

# Obter e imprimir a data atual
current_date = datetime.datetime.now()
print(f"Data atual: {current_date.strftime('%d/%m/%Y')}")
