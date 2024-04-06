import pandas as pd
from sklearn.datasets import load_iris

# Configurar o pandas para exibir todas as linhas e colunas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Mapeamento dos valores inteiros para os nomes das espécies
especies = {
    0: "Iris-Setosa",
    1: "Iris-Versicolour",
    2: "Iris-Virginica"
}

# Carregar o conjunto de dados Iris
iris = load_iris()

# Criar um DataFrame do Pandas com os dados e os rótulos de classe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Mapear os valores inteiros de destino de volta para os nomes das espécies
df['species'] = df['target'].map(especies)

# Exibir o número total de linhas
total_linhas = len(df)
print(f"O conjunto de dados Iris possui {total_linhas} linhas.\n")

# Exibir os dados na tela com formatação tabulada
print("Dados do conjunto de dados Iris:")
print(df.to_string(index=False))

print("\nOs dados foram exibidos acima.")
