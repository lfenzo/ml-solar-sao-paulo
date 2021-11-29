"""
ATTENTION!!

This script containg depracted code that SHOULD NOT BE USED AS PART OF THE PROJECT.
It was kept only as a record of the previous versions of the project during
development
"""

import os
import joblib
import argparse
import numpy as np

from datetime import datetime
from argparse import RawTextHelpFormatter


def fit_selected_model(model, X, Y, strategy: str, n_epochs, verbose = 0, param_grid = None, n_folds = 5):
    """
    Função responsável por treinar os modelos da classe do Scikit-Learn.

    Parâmetros
    -------------
    - `model`: Classe Sequential do Keras para ser treinada \\
    - `X`: Conjunto de dados para inferencia \\
    - `Y`: Variavel(is) alvo \\
    - `strategy`: Estratégia de trainamento: `'cross_validation'` ou `'grid_search'`. Caso o valor seja `'grid_search'`
    deverá ser especificado o espaço de busca por meio do atributo `param_grid`.
    - `n_epochs`: número de épocas para o treinamento do modelo.
    - `n_folds`: núemro de dobras que ocorrerão nos processos de cross-validation ou grid-search

    Retorna
    -------------
    Regressor `model` treinado
    """

    if strategy == 'cross_validation':

        from keras.wrappers.scikit_learn import KerasRegressor
        from sklearn.model_selection import cross_val_score

        # instancia um modelo que é empacotade para a API do scikit dessa forma é mais facil fazer o cross-validatoin. Não precisa implementar no braço
        scikit_wrapped_model = KerasRegressor(model, 
                                              epochs = n_epochs, 
                                              verbose = verbose,
                                              batch_size = 32)

        if verbose:
            print(scikit_wrapped_model)
            print('Iniciando o trainamento do modelo com cross-validation...')

        cross_validation = cross_val_score(scikit_wrapped_model, 
                                           X, 
                                           Y, 
                                           cv = n_folds, 
                                           scoring = 'r2',
                                           verbose = verbose)

        if verbose:
            print('Mostrando os scores após as etapas de cross-validation:')
            print(cross_validation)
        
        # model = cross_validation['estimator'][0] # ???

    elif strategy == 'fit':

        if verbose:
            print('Iniciando o método \'fit\' do modelo ...')

        from keras.callbacks import EarlyStopping

        model.fit(X, Y, 
                  epochs = n_epochs,
                  batch_size = 4,
                  validation_split = 0.1,
                  callbacks = [EarlyStopping(patience = 10)], 
                  verbose = verbose)

        if verbose:
            print('Treinamento do modelo completo!')

    elif strategy == 'grid_search':
        pass

        if param_grid is None:
            raise ValueError('Precisa especificar o espacço de busca válido. Voce está esquedendo de fazer isso')

        if verbose:
            print('Iniciando o grid-search no modelo ...')

        from sklearn.model_selection import GridSearchCV
    
        scikit_wrapped_model = KerasRegressor(model, 
                                              epochs = n_epochs, 
                                              verbose = verbose, 
                                              batch_size = 32)

        grid_search = GridSearchCV(estimator = scikit_wrapped_model, 
                                   param_grid = param_grid,
                                   cv = n_folds,
                                   verbose = verbose)
        
        grid_search.fit(X, Y)

        if verbose:
            print('Mostrando os melhores resultados dos modelos:')
            print(grid_search.best_params_)
            print(grid_search.best_score_)

        model = grid_search.best_estimator_

    return model

# ------------------------------------------------
# ------------------------------------------------

def save_model(model, arq, verbose = 0):
    """
    Salva o modelo especificado em `model` em memória persistente no formato: NOME_MODELO_timestamp.

    Parãmetros
    --------------
    - `model`: modelo treinado a ser salvo em disco.
    """

    from keras import models

    if arq == 'dnn':
        model_name = 'Deep_Neural_Network'
    elif arq == 'rnn':
        model_name = 'Recurrent_Neural_nNetwork'

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename = f'tf_{model_name}_{timestamp}.h5'

    if verbose:
        print(f'Salvando o modelo treinado no arquivo \'{filename}\' ...', end = ' ')

    models.save_model(model = model, filepath = filename)

    if verbose:
        print('feito!')

# ------------------------------------------------
# ------------------------------------------------

def choose_model(name: str, verbose: str):
    """
    Determina a arquitetura da rede neural a ser utilizada nos dados dado o parâmetro \'name\'.
    
    Retorna
    ---------
    Retorna um modelo com as quantidades de camadas ocultas 
    """

    from keras import Sequential
    from keras.layers import Dense, Flatten, Dropout, BatchNormalization

    if name == 'dnn':

        model = Sequential([
            Flatten(), 
            
            Dense(150, activation = 'selu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(150, activation = 'selu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(1)
        ])

        model.compile(loss = 'mse', optimizer = 'nadam')

    elif name == 'rnn':
        pass

    else:
        raise ValueError(f'Arquitetura {name} não está dispovível. Verifique e execute novamente.') 

    return model

# ------------------------------------------------
# ------------------------------------------------

if __name__ == '__main__':

    # o RawTextHelpFormatter permite que colocar quebras de linha, tab e outras coisas na ajuda
    ap = argparse.ArgumentParser(formatter_class = RawTextHelpFormatter)

    # =============================================================
    # ============== ESTRATEGIAS DE TRENAMENTO ====================
    # =============================================================

    metodos_treinamento = ap.add_mutually_exclusive_group(required = True)

    metodos_treinamento.add_argument('-cv', '--cross-validation', 
                                    action = 'store_true', 
                                    help='Treina o modelo especificado usando \'cross-validation\'.')

    metodos_treinamento.add_argument('-fit', '--standard-fit', 
                                    action='store_true', 
                                    help='Treina o modelo usando o método padrão \'fit\'.')

    metodos_treinamento.add_argument('-gs', '--grid-search', 
                                    action='store_true', 
                                    help='Executar o script usando Grid-Search e com cross-validation')

    # =============================================================
    # ====== ARQUITETURAS DE REDES NEURAIS DISPONÍVEIS ============
    # =============================================================

    # define qual sera a arquitetura utilizada
    ap.add_argument('-arq', '--nn-architecture',
                    metavar = '', 
                    required = True, 
                    type = str,
                    help = 'Arquitetura da Rede Neural a ser utilizada. Arquiteturas disponíveis:' +
                            '\n\t\'dnn\': Deep Neural Network' + 
                            '\n\t\'rnn\': Recurrent Neural Network')

    # =============================================================
    # =================== DEMAIS ARGUMENTOS =======================
    # =============================================================

    ap.add_argument('-n', '--n-folds',
                    metavar='', 
                    required = False, 
                    default = 5,
                    type = int,
                    help = 'Número de dobras para a realização do cross-validation. Default: 5.')

    ap.add_argument('-n_epcs', '--n-epochs',
                    metavar='', 
                    required = False, 
                    default = 5,
                    type = int,
                    help = 'Número de epocas para o treinamento do modelo. Default: 20.')

    ap.add_argument('-tst_pct', '--test-percentage',
                    metavar='', 
                    required = False, 
                    default = 15,
                    type = int,
                    help = 'Porcentagem do conjuunto de dados que é usada para a avaliação do modelo. Default: 15%%')

    ap.add_argument('-sm', '--save-model',
                    action = 'store_true',
                    default = True,
                    help = 'Salva o modelo em disco no formato [nome do modelo]_[timestamp].dat.\nO nome é definido aotmoaticamente de acordo com a classe do SKlearn utilizada. Default: True.')

    ap.add_argument('-ws', '--warm-start', 
                    metavar = '', 
                    required = False, 
                    default = 0,
                    type = int,
                    help = 'Comeca as computações a partir do cluster especificado.')

    ap.add_argument('-v', '--verbose', 
                    metavar='', 
                    required=False, 
                    default=0,
                    type = bool,
                    help='Mostrar informações conforme são feitas as operações. Default: 0')

    args = vars(ap.parse_args())

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------

    param_grid = None

    if args['cross_validation']:
        strategy = 'cross_validation'

    elif args['standard_fit']:
        strategy = 'fit'

    elif args['grid_search']:
        
        param_grid = {
            'n_estimators': np.arange(10, 200, 25)
        }
        
        strategy = 'grid_search'

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # -----------------------------------------------------------

    os.chdir('clusters')

    # gambiarra pra pegar os diretorios em ordem
    cluster_dirs = sorted([int(d) for d in os.listdir() if os.path.isdir(d)])
    cluster_dirs = list(map(str, [d for d in cluster_dirs]))

    if args['verbose']:
        print(f'Iniciando o treinamento de {len(cluster_dirs)} clusters com o algoritmo {args["nn_architecture"]}')

    for i, d in enumerate(cluster_dirs):

        # pula os clusters que estão antes do cluster especificado para o começo no 
        # argumento 'warm_start'
        if i < args['warm_start']:
            continue

        if args['verbose']:
            print(f'[{i:2} / {len(cluster_dirs) - 1}]\t', end = '')
            print(f'Iniciando o treainamento do cluster {d:2}', end = ' ')
        
        os.chdir(d)

        xtrain = np.load('x_train.npy')
        ytrain = np.load('y_train.npy')

        # instancia um modelo para cada uma dos clusters
        model = choose_model(name = args['nn_architecture'], verbose = args['verbose'])

        model = fit_selected_model(model = model, 
                                   X = xtrain, 
                                   Y = ytrain, 
                                   n_epochs = args['n_epochs'],
                                   strategy = strategy,
                                   n_folds = args['n_folds'], 
                                   verbose = args['verbose'], 
                                   param_grid = param_grid)
        
        if args['save_model']:
            save_model(model = model, 
                       arq = args['nn_architecture'], 
                       verbose = args['verbose'])
        
        os.chdir('..')
