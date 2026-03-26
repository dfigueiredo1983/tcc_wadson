# -*- coding: utf-8 -*-
"""Experimentos v11

#Trabalho de Conclusão de Curso
#Especialização em Ciência de Dados
Prof. Eduardo Kugler Viegas<BR>
Alunos: Humberto Pradera e Leonardo Rocha

Experimentos para subsidiar a construção do TCC<BR>

Utilizamos a versão NetFlow v3 Datasets

3 Objetivos:
- F1 DS2
- F1 DS3
- F1 DS4

Variáveis
- Quantidade de features
- Quantidade de neuronios
- Quantidade de camadas

Este Script tem a gravacao para poder rodar em ambiente cuja disponibilidade varie.

"""

# Imports, variáveis e funções gerais"""


#Bibliotecas
import numpy as np
import pandas as pd

import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#shrink memory dataset
from fastai.tabular.core import df_shrink

#CNN/MLP
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense,  Dropout
from tensorflow.keras.models import Sequential

#from google.colab import drive

from time import time as time2
import datetime
#import time
import os




# Controle de alguns experimentos
QUANTIDADE_MINIMA_CLASSE = 1000
HIGIENIZAR_DATASETS = True
BALANCEAR_DATASETS = True
TEST_SIZE = 0.2

# Define the expected path for the feature names file
LOG_DIR = os.path.expanduser('~/resultados/ds/')
LOG_EXECUCAO = 'exec.txt'
LOG_CHECKPOINT = 'checkpoint'
LOG_FINAL = 'final.pkl'

# Define o caminho do arquivo salvo
load_path = LOG_DIR+LOG_FINAL
#final_result_path = LOG_DIR+'nsga2_results_out.pkl'

#feature_names_path = LOG_DIR +'feature_names.txt'



# Verificação de GPU
print("--- Verificando disponibilidade de GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Configura o crescimento de memória dinâmico para evitar que o TensorFlow
    # aloque toda a memória da GPU de uma vez.
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"GPUs Físicas: {len(gpus)}, GPUs Lógicas: {len(logical_gpus)}")
    print("GPU disponível e configurada para uso.")
  except RuntimeError as e:
    # O crescimento de memória deve ser configurado antes da inicialização das GPUs
    print(e)
else:
    print("Nenhuma GPU encontrada. O treinamento será executado na CPU.")
print("----------------------------------------\n")

# --- Configurando a estratégia de distribuição para múltiplas GPUs ---
# print("--- Configurando a estratégia de distribuição ---")
# # Desativado para rodar em uma única GPU
# strategy = tf.distribute.MirroredStrategy()
# print(f"Estratégia de distribuição: MirroredStrategy com {strategy.num_replicas_in_sync} réplicas (GPUs).")
# print("----------------------------------------\n")

# Variavel que será redefinida posteriormente
attack_categories = ['Benigno', 'Ataque']

# Diretório para salvar os gráficos
#PLOT_DIR = os.path.expanduser('~/resultados/ds/plots')
#if not os.path.exists(PLOT_DIR):
#    os.makedirs(PLOT_DIR)

lista_campos_excluir = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
                        'DNS_QUERY_ID', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'MIN_TTL', 'MAX_TTL']
lista_campos_verif = ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES']

#Formata um número em bytes para Megabytes
def format_megabytes(size_in_bytes):
  if size_in_bytes is None:
    return "N/A"
  size_in_mb = size_in_bytes / (1024 * 1024)
  return f"{size_in_mb:.2f} MB"

def balacear_dataset(pdf):

  if not BALANCEAR_DATASETS:
    print("- Dataset não será balanceado")
    return pdf

  attack_counts = pdf['Attack'].value_counts()
  print("- attack_counts:",attack_counts,"\n")

  min_attack_count = attack_counts[attack_counts.index != 'Benign'].min()
  print("min_attack_count:",min_attack_count)

  benign_count = attack_counts['Benign']
  print("benign_count:", benign_count)

  # Calcula o valor alvo para as classes maliciosas
  target_non_benign_count = min(min_attack_count, benign_count // (len(attack_counts) - 1))
  print("- Quantidade inicial mínima para cada classe de ataque:",target_non_benign_count)

  if target_non_benign_count<QUANTIDADE_MINIMA_CLASSE:
    print(f"-- Então a quantidade inicial mínima é menor que {QUANTIDADE_MINIMA_CLASSE}. O Dataset terá problema no balanceamento.")
    # Se a quantidade inicial or menor que 1000, então teremos um Dataset balanceado com muito poucos casos.
    # Para evitar isto, busco a classe que possua menor representação que tenha quantidade de ocorrências maior que 1000
    # Pego a totalização por tipo de ataque
    vc = pdf['Attack'].value_counts()
    # Filtro de quantidade mínima, estipulada pelo professor como 1000
    vc1k = vc[vc>QUANTIDADE_MINIMA_CLASSE]
    # Pego o último valor, que deve ser o menor valor superior a 1000
    target_non_benign_count = vc1k.tail(1).values[0]
    print("-- Novo valor mínimo encontrado:",target_non_benign_count)

  # De fato faz o balanceamento
  balanced_dfs = []
  for attack_type in attack_counts.index:
    if attack_type == 'Benign':
      balanced_dfs.append(pdf[pdf['Attack'] == 'Benign'].sample(min(benign_count, target_non_benign_count * (len(attack_counts) - 1)), random_state=42))
    else:
      balanced_dfs.append(pdf[pdf['Attack'] == attack_type].sample(min(attack_counts[attack_type], target_non_benign_count), random_state=42))

  alldf = pd.concat(balanced_dfs)

  print("\nResultado do balanceamento:")
  attack_counts = alldf['Attack'].value_counts()
  print("- attack_counts:",attack_counts,"\n")

  return alldf

def higienizar_dataset(pdf):

  if not HIGIENIZAR_DATASETS:
    print("- Dataset não será higienizado")
    return

  qtd_ini = len(pdf)
  print("- Tamanho original do Dataset: {:,}".format(qtd_ini))

  # Drop rows where SRC_TO_DST_SECOND_BYTES or DST_TO_SRC_SECOND_BYTES have infinite values or values > np.finfo(np.float64).max
  print("- Removendo registros com campos com valor infinito ou superior a np.float64")
  for col in lista_campos_verif:
    pdf.drop(pdf[np.isinf(pdf[col])].index, inplace=True)
    pdf.drop(pdf[pdf[col] > np.finfo(np.float64).max].index, inplace=True)

  #retira registros com campos NA
  print("- Removendo registros com campos NA")
  pdf.dropna(subset=['SRC_TO_DST_SECOND_BYTES'], inplace=True)

  # verificando se há linhas duplicadas
  qtd_duplicadas = pdf.duplicated().sum()

  if qtd_duplicadas > 0:
    print("- Encontrados {:,} registros duplicados".format(qtd_duplicadas))
    # retirando as linhas duplicadas
    pdf.drop_duplicates(inplace=True)
    pdf.reset_index(inplace=True, drop=True)

  qtd_fim = len(pdf)
  print("- Ao total, foram eliminados {:,} registros".format(qtd_ini-qtd_fim))
  print("- Tamanho final {:,}".format(qtd_fim))


# Datasets ###############################################
#drive.mount('/content/gdrive/', force_remount=True)

df_url = []
df = []

#df_url.append('~/datasets/ds/NF-UNSW-NB15-v3.csv')
df_url.append('#')
df_url.append('~/datasets/ds/NF-ToN-IoT-v3.csv')
df_url.append('~/datasets/ds/NF-BoT-IoT-v3.csv')
df_url.append('~/datasets/ds/NF-CICIDS2018-v3.csv')

tamanho_antes = 0
tamanho_depois = 0

#limita a quantidade de registros a carregar ao mesmo tempo em memóris
#para permitir uma máquina com mesmo tempo processar os Datasets
chunk_size = 4000000
chunk_resultset = []

#loop carregando os datasets
for index, d in enumerate(df_url):
  #carrega
  print("\n--- Dataset",index+1,'---')
  if d == '#':
    print("Dataset desativado para processamento.")
    # Para manter compatibilidade com o restante do código, adiciona um DataFrame vazio
    df.append(pd.DataFrame())
    continue

  #cria um iterator para ler o arquivo em pedaços
  chunk_iter = pd.read_csv(d, chunksize=chunk_size)

  chunk_number = 0
  for chunk in chunk_iter:
    chunk_number += 1
    print(f"---- Processando chunk {chunk_number} ----")
    print("- Lidas {:,} linhas.".format(len(chunk)))

    tamanho_memoria_ds = chunk.memory_usage(deep=True).sum()
    print("Chunk carregado:",format_megabytes(tamanho_memoria_ds))
    tamanho_antes += tamanho_memoria_ds

    #elimina colunas desnecessárias
    chunk = chunk.drop(lista_campos_excluir, axis=1)
    print("Campos desnecessários removidos.")

    #higieniza dataset
    higienizar_dataset(chunk)
    print("Chunk higienizado.")

    chunk_resultset.append(chunk)
    gc.collect()
    print("Garbage collector invocado para o chunk")

  #unifica o que restou dos chunks
  mydf = pd.concat(chunk_resultset)
  chunk_resultset.clear()

  #Só para garantir que não teremos duplicados resultantes da união dos chunks
  higienizar_dataset(mydf)
  print("Dataset higienizado.")

  #balenceia o dataset
  mydf = balacear_dataset(mydf)
  print("Dataset balanceado.")

  #reduz o tamanho
  mydf = df_shrink(mydf)
  print("Dataset reduzido.")
  print("- Tamanho atual: {:,} linhas.".format(len(mydf)))

  gc.collect()
  print("Garbage Collector invocado para o final.")

  tamanho_memoria_ds = mydf.memory_usage(deep=True).sum()
  tamanho_depois += tamanho_memoria_ds

  #anexa na lista de datasets
  df.append(mydf)
  del mydf # Libera a memória do DataFrame unificado após ser adicionado à lista
  gc.collect()

print(" ")
print("Tamanho original dos Datasets:",format_megabytes(tamanho_antes))
print("Tamanho final dos Datasets:",format_megabytes(tamanho_depois))

print("Linhas lidas dos DS:")
for i, d in enumerate(df):
  print(f"DS{i+1}:",len(d))

print("Unifica os 3 Datasets para análise conjunta.")
df_all = pd.concat([df[1], df[2], df[3]])
print(f"DF Unificado:",len(df_all))


# Estrutura dos Datasets
# TODOS OS DATASETS possuem a mesma estrutura

# Definição dos conjuntos de dados
# Conjuntos de dados DS1 ###############################################
#DESATIVADO PARA NÂO SER POSSÌVEL RODAR - DATASET A SER DESCARTADO


### Conjunto de dados DS2 ###############################################
X2 = df[1].drop(['Attack', 'Label'], axis=1)
y2 = df[1].loc[:, 'Label']


### Conjunto de dados DS3 ###############################################
X3 = df[2].drop(['Attack', 'Label'], axis=1)
y3 = df[2].loc[:, 'Label']


### Conjunto de dados DS4 ###############################################
X4 = df[3].drop(['Attack', 'Label'], axis=1)
y4 = df[3].loc[:, 'Label']


### Conjunto de dados DS ALL ###############################################
XAll = df_all.drop(['Attack', 'Label'], axis=1)
yAll = df_all.loc[:, 'Label']


# FUNÇÂO DE CRIAÇÂO DE MODELO ###############################################

def cria_modelo(pShape, pOutput=True, pQtdNeurons=5, pQtdCamadas=1):
  # Validacao dos parâmetros
  if (pQtdCamadas < 1) or (pQtdCamadas > 9):
    raise ValueError("O número de camadas deve ser maior do que 0 e inferior a 10.")
  
  ## Modelo base
  modelo = Sequential([
      # Camada de Entrada
      Input(shape=(pShape,))
  ])

  # Adiciona camadas ocultas e Dropout baseadas em pQtdCamadas
  for _ in range(pQtdCamadas):
      modelo.add(Dense(2**pQtdNeurons, activation='relu'))
      # Dropout para prevenir overfitting
      modelo.add(Dropout(0.1))

  # Camada de saída
  modelo.add(Dense(2, activation='softmax'))

  # Arquitetura
  if pOutput:
    modelo.summary()
    print(f"Modelo criado com {pQtdCamadas} camadas ocultas e {2**pQtdNeurons} neurônios por camada.")
    print(f"Shape de entrada: {pShape}")

  ## Compilando o modelo
  if pOutput:
    print("\nCompilando modelo...")

  modelo.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )

  if pOutput:
    print(f"Learning rate: {modelo.optimizer.learning_rate.numpy()}")
    print("Compilação completa.\n")

  return modelo

##################################################################################
# MultiObjetivo Optimization com Pymoo
# !pip install -U pymoo

from pymoo.algorithms.moo.nsga2 import NSGA2 
from pymoo.optimize import minimize

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

import pickle



###############################################################################################
# Custom Callback for saving checkpoints
###############################################################################################
class CheckpointCallback(Callback):

    def __init__(self, filename="checkpoint.pkl", verbose=False):
        super().__init__()
        self.filename = filename
        self.verbose = verbose
        self.n_eval = 0

    def notify(self, algorithm):
        # Save the algorithm state
        s = f"{self.filename}_{algorithm.n_gen}.pkl"
        with open(s, 'wb') as f:
            pickle.dump(algorithm, f)

        if self.verbose:
            print(f"Checkpoint saved at generation {algorithm.n_gen} to {s}")

        # reduz o consumo de memória a cada geração
        gc.collect()
        if self.verbose:
            print(f"Garbage collection completed at generation {algorithm.n_gen}")


def Minimize(problem):

    #inicialização padrão, com poucas features
    #amostragem_customizada = SamplingComFeaturesFixas(n_features_para_selecionar=5)

    # Define the path for saving and loading the checkpoint
    checkpoint_path = LOG_DIR + LOG_CHECKPOINT

    # Create the checkpoint callback
    checkpoint_callback = CheckpointCallback(filename=checkpoint_path, verbose=True)

    # Check if a checkpoint exists and load it
    if os.path.exists(f"{checkpoint_path}.pkl"):
        print(f"Loading checkpoint from: {checkpoint_path}.pkl")

        with open(f"{checkpoint_path}.pkl", 'rb') as f:
            algorithm = pickle.load(f)
        print(f"Resuming from generation {algorithm.n_gen}")

        # Sincroniza a contagem de avaliações do problema com o estado do algoritmo carregado.
        # O algoritmo já sabe quantas avaliações foram feitas até o último checkpoint.
        # Isso evita a recontagem de indivíduos já avaliados em gerações anteriores.
        problem.contagem = algorithm.evaluator.n_eval

        # Continua otimização
        res = minimize(problem,
                       algorithm, # passa o algoritmo carregado
                       seed=42,   # mantém o mesmo seed original
                       save_history=True,
                       verbose=True,
                       termination=('n_gen', 100),  # continua até o número de gerações chega a 100
                       callback=checkpoint_callback # callback para gravação
                       )
    else:
        print(f"No checkpoint found with name {checkpoint_path}.pkl. Starting new optimization.")
       
        algorithm = NSGA2(pop_size=100)

        #print(f"Algorithm Crossover: {algorithm.mating.crossover}")
        #print(f"Algorithm Mutação: {algorithm.mating.mutation}")

        res = minimize(problem,
                       algorithm,
                       seed=42,
                       save_history=True,
                       verbose=True,
                       termination=('n_gen', 100),
                       callback=checkpoint_callback # callback para gravação
                       )

    # Save the final result
    final_result_path = LOG_DIR + LOG_FINAL
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)

    with open(final_result_path, 'wb') as f:
        pickle.dump(res, f)
    print(f"Final result saved to: {final_result_path}")


    return res

###############################################################################################
# Classe do problema a ser resolvido
###############################################################################################
class NetFlowProblem(ElementwiseProblem):

    def __init__(self, *args):
        self.contagem = 0
        self.total_tempo = 0
        self.modelAll = None

        # 🔥 CACHE DE MEMOIZAÇÃO
        self.cache = {}

        super().__init__(
            n_var=42,
            n_obj=3,
            xl=np.concatenate((np.zeros(40), [2,1])),
            xu=np.concatenate((np.ones(40) , [9,5])),
            n_ieq_constr=3
        )

    def is_to_delete_feature(self, prob):
        return prob < 0.5

    def adjust_features(self, features_bin, X_features):
        delete_indices = [i for i, v in enumerate(features_bin) if v == 0]
        cols_to_delete = X_features.columns[delete_indices]
        return X_features.drop(columns=cols_to_delete)

    def create_and_train_model(self, pX, py, pQtdNeurons, pQtdCamadas):

        x_train, x_test, y_train, y_test = train_test_split(
            pX, py, test_size=0.2, random_state=42, stratify=py
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        mymodel = cria_modelo(
            pShape=x_train_scaled.shape[1],
            pOutput=False,
            pQtdNeurons=pQtdNeurons,
            pQtdCamadas=pQtdCamadas
        )

        mymodel.fit(
            x_train_scaled,
            y_train,
            epochs=15,
            batch_size=64,
            validation_data=(x_test_scaled, y_test),
            verbose=0
        )

        del x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler
        gc.collect()

        return mymodel

    def get_predicted_class(self, pX, py):

        x_train, x_test, y_train, y_test = train_test_split(
            pX, py, test_size=0.2, random_state=42, stratify=py
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        predictions_prob = self.modelAll.predict(x_test_scaled, verbose=0)
        predictions = np.argmax(predictions_prob, axis=1)

        del x_train, x_test, y_train, x_train_scaled, x_test_scaled, predictions_prob
        gc.collect()

        return predictions, y_test

    def evaluate_model_f1(self, pX, py):

        y_pred, y_true = self.get_predicted_class(pX, py)

        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
            epsilon = 1e-7
            tpr = tp / (tp + fn + epsilon)
            tnr = tn / (tn + fp + epsilon)
        except ValueError:
            tpr, tnr = 0.0, 0.0

        return (1 - f1), tpr, tnr

    def _evaluate(self, x, out, *args, **kargs):

        self.contagem += 1

        # 🔥 DISCRETIZA FEATURES
        features_bin = tuple((x[:-2] >= 0.5).astype(int))

        num_neurons = int(x[-2])
        qtd_camadas = int(x[-1])

        # 🔥 CHAVE DE CACHE
        cache_key = (features_bin, num_neurons, qtd_camadas)

        # 🔥 VERIFICA CACHE
        if cache_key in self.cache:
            print(f"Avaliação {self.contagem:05d} (REUTILIZADA CACHE)")

            f1_2, f1_3, f1_4 = self.cache[cache_key]
            out["F"] = [f1_2, f1_3, f1_4]
            return

        t0 = time2()
        print(f"Avaliação {self.contagem:05d}", end="")

        f1_2 = f1_3 = f1_4 = 1.0

        # Dataset ALL
        X_tmp = XAll.copy()
        y_tmp = yAll.copy()

        X_tmp = self.adjust_features(features_bin, X_tmp)
        qtd_features = len(X_tmp.columns)

        parametros_validos = True

        if qtd_features < 2 or num_neurons < 2:
            parametros_validos = False

        if qtd_camadas < 1 or qtd_camadas > 5:
            parametros_validos = False

        if parametros_validos:

            self.modelAll = self.create_and_train_model(
                X_tmp, y_tmp, num_neurons, qtd_camadas
            )

            # DS2
            X_tmp = self.adjust_features(features_bin, X2.copy())
            f1_2, _, _ = self.evaluate_model_f1(X_tmp, y2.copy())

            # DS3
            X_tmp = self.adjust_features(features_bin, X3.copy())
            f1_3, _, _ = self.evaluate_model_f1(X_tmp, y3.copy())

            # DS4
            X_tmp = self.adjust_features(features_bin, X4.copy())
            f1_4, _, _ = self.evaluate_model_f1(X_tmp, y4.copy())

            # limpeza
            del self.modelAll
            self.modelAll = None
            tf.keras.backend.clear_session()

        # 🔥 SALVA NO CACHE
        self.cache[cache_key] = (f1_2, f1_3, f1_4)

        t1 = time2() - t0
        print(f", duração: {t1:.2f}", end="")

        out["F"] = [f1_2, f1_3, f1_4]

        print(f" | F1 DS2:{1-f1_2:.4f}, DS3:{1-f1_3:.4f}, DS4:{1-f1_4:.4f}")

        gc.collect()


################################################ Fim da classe do problema ##########################################################

problem = NetFlowProblem()

### Execução do pymoo

res_dt = Minimize(problem)

print("Processamento concluído.")
