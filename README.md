English description below:


Predição de Diabetes com Modelos de Machine Learning

Sobre o Projeto

Neste projeto, desenvolvi um modelo de aprendizado de máquina para prever os níveis de diabetes com base no conjunto de dados Diabetes do sklearn.datasets. 

Utilizei diferentes abordagens, incluindo redes neurais artificiais e modelos tradicionais de regressão, e implementei um modelo de Stacking para combinar as previsões de diferentes algoritmos.

O que eu fiz

1. Carregamento e Preparação dos Dados

Utilizei o conjunto de dados load_diabetes() da biblioteca sklearn.

Extraí as features (X) e os rótulos (y).

Normalizei os dados com StandardScaler para garantir que todas as features estivessem na mesma escala.

Dividi os dados em treino e teste usando train_test_split.

2. Construção e Treinamento de uma Rede Neural

Criei um modelo Sequential do Keras com:

3 camadas densas de 300 neurônios com ativação ReLU.

Camadas de Dropout (40%) para evitar overfitting.

Uma camada de saída para previsão numérica.

Compilei o modelo utilizando:

Otimizador Nadam.

Função de perda MSE (Erro Quadrático Médio).

Métrica de avaliação MAE (Erro Absoluto Médio).

Treinei o modelo por 2000 épocas, salvando apenas o melhor modelo utilizando ModelCheckpoint.

3. Avaliação e Predição

Carreguei o melhor modelo salvo e avaliei sua performance no conjunto de teste.

4. Implementação de um Modelo de Stacking

Combinei RandomForestRegressor, SVR e LinearRegression como modelos base.

Utilize um MLPRegressor com:

4 camadas ocultas (300, 300, 200, 100 neurônios) e ativação ReLU.

adam como otimizador.

learning_rate adaptativo.

Treinei o modelo StackingRegressor.

Testei as previsões do modelo e comparei com os valores reais.


-------------------------------------------------------------------------------------------------

Diabetes Prediction with Machine Learning Models

About the Project

In this project, I developed a machine learning model to predict diabetes levels based on the Diabetes dataset from sklearn.datasets.

I utilized different approaches, including artificial neural networks and traditional regression models, and implemented a Stacking model to combine predictions from various algorithms.

What I Did

1. Data Loading and Preparation

I used the load_diabetes() dataset from the sklearn library.

I extracted the features (X) and labels (y).

I normalized the data using StandardScaler to ensure all features were on the same scale.

I divided the data into training and testing sets using train_test_split.

2. Neural Network Construction and Training

I built a Sequential model in Keras with:

Three dense layers of 300 neurons using ReLU activation.

Dropout layers (40%) to prevent overfitting.

An output layer for numerical prediction.

I compiled the model using:

Nadam optimizer.

MSE (Mean Squared Error) as the loss function.

MAE (Mean Absolute Error) as the evaluation metric.

I trained the model for 2000 epochs, saving only the best model using ModelCheckpoint.

3. Evaluation and Prediction

I loaded the best saved model and evaluated its performance on the test set.

4. Implementation of a Stacking Model

I combined RandomForestRegressor, SVR, and LinearRegression as base models.

I used an MLPRegressor as the meta-model with:

Four hidden layers (300, 300, 200, 100 neurons) using ReLU activation.

Adam optimizer.

Adaptive learning rate.

I trained the StackingRegressor.

I tested model predictions and compared them with actual values.
