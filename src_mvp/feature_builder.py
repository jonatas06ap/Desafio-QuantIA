# feature_builder.py
#
# Módulo da Fase 3: Contém as funções de engenharia de features.
# Este arquivo não deve ser executado diretamente.
# Ele será importado pelo seu Jupyter Notebook.

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuração de Paths (usados pelas funções) ---
PATH_RAW = 'data/raw'
PATH_PROCESSED = 'data/processed'


# ==============================================================================
# FUNÇÃO 1: Extração de Sentimento
# ==============================================================================

def extrair_sentimento_pre_calculado(arquivo_json_entrada, pasta_saida=PATH_PROCESSED):
    """
    Lê um arquivo JSON de notícias (do EventRegistry), extrai o 
    score de sentimento JÁ EXISTENTE e salva um DataFrame em CSV.
    """
    
    print(f"Iniciando Etapa 1: Extraindo sentimento de '{arquivo_json_entrada}'...")
    
    try:
        with open(arquivo_json_entrada, 'r', encoding='utf-8') as f:
            data = json.load(f)
        articles = data.get('articles', {}).get('results', []) 
        if not articles:
            print("Nenhum artigo encontrado no arquivo JSON.")
            return None
        print(f"Carregados {len(articles)} artigos.")
    except Exception as e:
        print(f"Erro ao ler o arquivo JSON: {e}")
        return None

    resultados = []
    for article in tqdm(articles, desc="Extraindo Sentimento"): 
        score = article.get('sentiment')
        timestamp = article.get('dateTime')
        titulo = article.get('title', '')
        
        if score is None or not timestamp:
            continue
            
        resultados.append({
            'timestamp': timestamp,
            'sentiment_score': float(score),
            'title': titulo
        })
            
    if not resultados:
        print("Nenhum sentimento pôde ser extraído (Verifique 'includeArticleSentiment: True').")
        return None

    df_sentimento = pd.DataFrame(resultados)
    df_sentimento['datetime'] = pd.to_datetime(df_sentimento['timestamp'])
    df_sentimento = df_sentimento.set_index('datetime').sort_index()
    
    os.makedirs(pasta_saida, exist_ok=True)
    nome_arquivo = "sentimento_noticias.csv"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    
    df_sentimento.to_csv(caminho_arquivo)
    print(f"\nEtapa 1 Concluída. Resultados salvos em: {caminho_arquivo}")
    
    return df_sentimento

# ==============================================================================
# FUNÇÃO 2: Indicadores Técnicos
# ==============================================================================

def calcular_indicadores_tecnicos(arquivo_csv_entrada, pasta_saida=PATH_PROCESSED):
    """
    Lê um arquivo CSV de preços OHLCV e calcula indicadores técnicos.
    """
    
    print(f"\nIniciando Etapa 2: Calculando indicadores de '{arquivo_csv_entrada}'...")
    
    try:
        df = pd.read_csv(arquivo_csv_entrada, index_col='datetime', parse_dates=True)
        df.sort_index(inplace=True) 
        print(f"Carregados {len(df)} registros de preços.")
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV de preços: {e}")
        return None

    # Calcular Indicadores
    df['ma_curta'] = df['close'].rolling(window=10).mean()
    df['ma_longa'] = df['close'].rolling(window=50).mean()
    
    periodo_rsi = 14
    delta = df['close'].diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(window=periodo_rsi).mean()
    media_perda = perda.rolling(window=periodo_rsi).mean()
    rs = media_ganho / media_perda
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['momentum_5p'] = df['close'].pct_change(periods=5)
    df['volatilidade_20p'] = df['close'].pct_change().rolling(window=20).std()

    df_com_features = df.dropna()
    
    if df_com_features.empty:
        print("Erro: Nenhum dado restou após o cálculo dos indicadores.")
        return None
        
    print(f"Indicadores calculados. {len(df_com_features)} registros válidos restantes.")
    
    os.makedirs(pasta_saida, exist_ok=True)
    nome_base_arquivo = os.path.basename(arquivo_csv_entrada).replace('ohlcv_', '')
    nome_arquivo_saida = f"features_precos_{nome_base_arquivo}"
    caminho_arquivo_saida = os.path.join(pasta_saida, nome_arquivo_saida)
    
    df_com_features.to_csv(caminho_arquivo_saida)
    print(f"Etapa 2 Concluída. Features de preço salvas em: {caminho_arquivo_saida}")
    
    return df_com_features

# ==============================================================================
# FUNÇÃO 3: Criação do Dataset Final
# ==============================================================================

def criar_dataset_final(arquivo_features_precos, arquivo_sentimento, timeframe_precos, pasta_saida=PATH_PROCESSED):
    """
    Junta os dados de preço (com indicadores) e os dados de sentimento
    para criar o dataset final de treinamento.
    """
    
    print(f"\nIniciando Etapa 3: Criando dataset final...")
    
    try:
        df_precos = pd.read_csv(arquivo_features_precos, index_col='datetime', parse_dates=True)
        df_sentimento = pd.read_csv(arquivo_sentimento, index_col='datetime', parse_dates=True)
        
        if df_precos.index.tz is None:
            df_precos.index = df_precos.index.tz_localize('UTC')
        df_sentimento.index = df_sentimento.index.tz_convert('UTC')
            
        print("Arquivos de features de preço e sentimento carregados e padronizados para UTC.")
        
    except Exception as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None

    # Processar Sentimento
    print(f"Agregando sentimento para o timeframe: '{timeframe_precos}'")
    df_sentimento_agregado = df_sentimento['sentiment_score'].resample(timeframe_precos).mean()
    df_sentimento_agregado = df_sentimento_agregado.fillna(0.0)
    df_sentimento_agregado = df_sentimento_agregado.to_frame(name='sentimento_medio')

    # Juntar DataFrames
    df_final = df_precos.join(df_sentimento_agregado, how='left')
    df_final['sentimento_medio'] = df_final['sentimento_medio'].fillna(0.0)

    # Criar a Coluna Alvo
    periodo_previsao = 1
    df_final['target_preco_futuro'] = df_final['close'].shift(-periodo_previsao)
    df_final['target_preco_subiu'] = (df_final['target_preco_futuro'] > df_final['close']).astype(int)
    
    # Limpeza Final
    df_final = df_final.dropna()
    df_final = df_final.drop(columns=['target_preco_futuro'])
    
    print("Dataset final unido e coluna 'target' criada.")
    
    # Salvar em Parquet
    os.makedirs(pasta_saida, exist_ok=True)
    nome_arquivo_saida = "dataset_final_treinamento.parquet"
    caminho_arquivo_saida = os.path.join(pasta_saida, nome_arquivo_saida)
    
    df_final.to_parquet(caminho_arquivo_saida)
    
    print(f"\nEtapa 3 Concluída. Dataset final salvo em: {caminho_arquivo_saida}")
    
    return df_final