# feature-builder.py

"""
Módulo de Engenharia de Features (Fase 3)

Este script contém as funções necessárias para processar os dados brutos
(OHLCV e Notícias com sentimento 'bag of words') e transformá-los em um
dataset pronto para o treinamento de modelos de Machine Learning.

Funções principais:
- calculate_technical_indicators: Adiciona colunas de indicadores técnicos.
- aggregate_sentiment_features: Agrega dados de notícias na granularidade desejada.
- create_feature_dataset: Orquestra o pipeline completo de criação de features.

Pode ser executado diretamente para gerar um arquivo CSV de treino:
$ python feature-builder.py dados_ohlcv.csv dados_noticias.csv treino_final.csv
"""
import json
import pandas as pd
import pandas_ta as ta
import sys
import argparse  # Usado para uma interface de linha de comando limpa

def calculate_technical_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula e anexa indicadores técnicos ao DataFrame OHLCV.

    Justificativas:
    - RSI: Mede momentum de sobrecompra/sobrevenda.
    - MACD: Identifica momentum de tendência e seus cruzamentos.
    - BBands: Mede a volatilidade e níveis de preço "extremos".
    - ATR: Mede a volatilidade do mercado (útil para risco).
    """
    print("Calculando indicadores técnicos...")
    # Usamos .copy() para evitar o SettingWithCopyWarning
    df = df_ohlcv.copy()
    
    # 1. RSI (Relative Strength Index)
    df.ta.rsi(length=14, append=True)

    # 2. MACD (Moving Average Convergence Divergence)
    # Adiciona MACD, Histograma (MACDh) e Sinal (MACDs)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # 3. Bandas de Bollinger (Bollinger Bands)
    # Adiciona Lower, Mid, Upper, Bandwidth, Percent
    df.ta.bbands(length=20, std=2, append=True)

    # 4. ATR (Average True Range)
    df.ta.atr(length=14, append=True)
    
    # Remove as linhas iniciais que não tiveram dados suficientes para os cálculos
    df = df.dropna()
    return df

def aggregate_sentiment_features(df_news: pd.DataFrame, granularity: str = 'D') -> pd.DataFrame:
    """
    Agrega dados de notícias (sentimento e volume) na granularidade de tempo especificada.

    Justificativas:
    - sentiment_mean: Captura o "clima" geral do mercado, suavizando o ruído de notícias individuais.
    - news_volume: Mede a "atenção" do mercado. Picos de volume são indicadores de volatilidade.
    """
    print(f"Agregando features de sentimento na granularidade: {granularity}")
    
    # Resample(granularity) agrupa o índice de timestamp (ex: 'D' = Diário)
    df_agg = df_news.resample(granularity).agg(
        # 'sentiment' é o seu campo 'bag of words'
        sentiment_mean=('sentiment', 'mean'), 
        news_volume=('sentiment', 'count')
    )
    
    # Períodos sem notícias terão NaN. Preenchemos a média com 0 (neutro).
    # O volume já será 0 (pela contagem), mas o fillna(0) garante
    df_agg['sentiment_mean'] = df_agg['sentiment_mean'].fillna(0)
    
    return df_agg

def create_feature_dataset(df_ohlcv: pd.DataFrame, df_news: pd.DataFrame, 
                           granularity: str = 'D', 
                           include_target: bool = True, 
                           periods_future: int = 1) -> pd.DataFrame:
    """
    Orquestra o pipeline completo de criação de features.
    
    1. Calcula indicadores técnicos do OHLCV.
    2. Agrega features de sentimento das notícias.
    3. Une os dois DataFrames.
    4. (Opcional) Cria a variável alvo (target) para o treinamento.
    """
    
    # Etapa 1: Features Técnicas
    df_tech = calculate_technical_indicators(df_ohlcv)
    
    # Etapa 2: Features de Sentimento
    df_sentiment_agg = aggregate_sentiment_features(df_news, granularity)
    
    # Etapa 3: Unir os datasets
    # .join() alinha perfeitamente pelos índices de timestamp
    print("Unindo datasets de preço e sentimento...")
    df_merged = df_tech.join(df_sentiment_agg)
    
    # Lida com NaNs pós-junção
    # Se houver períodos de preço (ex: fim de semana) sem notícias,
    # preenchemos com o último valor de sentimento/volume conhecido.
    df_merged['sentiment_mean'] = df_merged['sentiment_mean'].fillna(method='ffill')
    df_merged['news_volume'] = df_merged['news_volume'].fillna(method='ffill')
    
    # Se ainda houver NaNs no *início* (antes da primeira notícia), preenche com 0
    df_merged = df_merged.fillna(0)
    
    # Etapa 4: (Opcional) Criação do Alvo
    if include_target:
        print(f"Criando variável alvo (target) para {periods_future} período(s) à frente...")
        # Pega o preço de fechamento 'N' períodos no futuro
        df_merged['future_close'] = df_merged['close'].shift(-periods_future)
        
        # O alvo é 1 se o preço futuro for maior que o atual, 0 caso contrário
        df_merged['target'] = (df_merged['future_close'] > df_merged['close']).astype(int)
        
        # Remove a coluna 'future_close' (data leakage) e os NaNs criados pelo shift
        df_merged = df_merged.drop(columns=['future_close'])
        df_merged = df_merged.dropna()
    
    print("Dataset de features criado com sucesso.")
    return df_merged

def load_data_from_files(ohlcv_path: str, news_path: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Carrega e prepara os arquivos brutos, padronizando ambos para fuso horário UTC (tz-aware).
    - OHLCV (CSV): Espera 'datetime' (assume ser UTC naive) e o localiza.
    - Notícias (JSON): Espera 'dateTimePub' (com fuso) e o converte para UTC.
    """
    
    # --- 1. Carregar Dados OHLCV (CSV) ---
    print(f"Carregando dados OHLCV de: {ohlcv_path}")
    df_ohlcv = pd.read_csv(ohlcv_path)
    
    # VERIFICA e RENOMEIA a coluna 'datetime' <--- CORRIGIDO
    if 'datetime' not in df_ohlcv.columns:  # <--- CORRIGIDO
        raise KeyError(f"Erro: Coluna 'datetime' não encontrada em {ohlcv_path}. Colunas disponíveis: {df_ohlcv.columns.tolist()}")
    
    # Renomeia para o nosso nome padrão 'timestamp'
    df_ohlcv = df_ohlcv.rename(columns={'datetime': 'timestamp'}) # <--- CORRIGIDO
    
    # Processamento padrão
    df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])
    
    # --- !! CORREÇÃO DO FUSO HORÁRIO !! ---
    try:
        df_ohlcv['timestamp'] = df_ohlcv['timestamp'].dt.tz_localize('UTC')
    except TypeError:
        df_ohlcv['timestamp'] = df_ohlcv['timestamp'].dt.tz_convert('UTC')
    # --- FIM DA CORREÇÃO ---

    df_ohlcv = df_ohlcv.set_index('timestamp').sort_index()
    df_ohlcv.columns = [col.lower() for col in df_ohlcv.columns]


    # --- 2. Carregar Dados de Notícias (JSON Aninhado) ---
    print(f"Carregando dados de Notícias (JSON aninhado) de: {news_path}")
    df_news = None
    try:
        # Abre o arquivo JSON como um dicionário Python
        with open(news_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 'Achatamos' a estrutura aninhada
        # data['articles']['results'] <--- CORRIGIDO
        df_news = pd.json_normalize(data, record_path=['articles', 'results']) # <--- CORRIGIDO
        
    except Exception as e:
        print(f"Erro ao tentar carregar o JSON aninhado (articles.results): {e}", file=sys.stderr) # <--- CORRIGIDO
        print("Verifique se o caminho do arquivo está correto e se a estrutura é { 'articles': { 'results': [...] } }", file=sys.stderr) # <--- CORRIGIDO
        raise

    if 'dateTimePub' not in df_news.columns:
        raise KeyError(f"Erro: Coluna 'dateTimePub' não encontrada no JSON.")

    df_news = df_news.rename(columns={'dateTimePub': 'timestamp'})
    
    # Processamento padrão
    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])
    
    # --- !! CORREÇÃO DO FUSO HORÁRIO !! ---
    df_news['timestamp'] = df_news['timestamp'].dt.tz_convert('UTC')
    # --- FIM DA CORREÇÃO ---

    df_news = df_news.set_index('timestamp').sort_index()
    
    return df_ohlcv, df_news