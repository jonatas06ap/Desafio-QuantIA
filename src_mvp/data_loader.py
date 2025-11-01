import ccxt
import pandas as pd
import os
from datetime import datetime
import time
import json
from newsapi import NewsApiClient
from dotenv import load_dotenv

def get_historical_ohlcv(ativo, timeframe, data_inicio_str, pasta_saida='data/raw'):
    """
    Busca dados históricos (OHLCV) de um ativo usando ccxt e salva em CSV.

    Lida automaticamente com a paginação para buscar todo o histórico
    desde a data de início.

    Parâmetros:
    - ativo (str): O ticker do ativo (ex: 'BTC/USDT', 'ETH/USDT').
    - timeframe (str): O timeframe desejado (ex: '1h', '4h', '1d').
    - data_inicio_str (str): Data de início no formato 'YYYY-MM-DD'.
    - pasta_saida (str): Pasta onde o CSV será salvo.
    """
    
    # 1. Inicializa a exchange (Binance)
    exchange = ccxt.binance({
        'rateLimit': 1200,  # Respeita o limite da API
        'enableRateLimit': True
    })

    print(f"Buscando dados para {ativo} no timeframe {timeframe} desde {data_inicio_str}...")

    # 2. Converte a data de início para timestamp em milissegundos (necessário pelo ccxt)
    dt_inicio_ms = exchange.parse8601(f"{data_inicio_str}T00:00:00Z")

    # 3. Loop de Paginação para buscar todos os dados
    limite_por_chamada = 1000  # Limite comum da Binance
    todos_os_dados = []
    
    while True:
        try:
            # Busca os dados OHLCV
            novos_dados = exchange.fetch_ohlcv(
                ativo, 
                timeframe, 
                since=dt_inicio_ms, 
                limit=limite_por_chamada
            )
            
            # Se não retornar dados novos, saímos do loop
            if not novos_dados:
                break
                
            print(f"  > Recebidos {len(novos_dados)} velas...")
            
            # Adiciona os dados novos à nossa lista
            todos_os_dados.extend(novos_dados)
            
            # Atualiza o 'since' para a próxima chamada
            # Pega o timestamp da ÚLTIMA vela + 1 milissegundo
            dt_inicio_ms = novos_dados[-1][0] + 1
            
            # Pausa para respeitar o rate limit da API
            time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            print(f"Erro de rede: {e}. Tentando novamente em 5s...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"Erro da exchange: {e}. Abortando.")
            return None
        except Exception as e:
            print(f"Erro inesperado: {e}. Abortando.")
            return None

    if not todos_os_dados:
        print(f"Nenhum dado encontrado para {ativo}.")
        return None

    # 4. Converte a lista de dados em um DataFrame do Pandas
    df = pd.DataFrame(todos_os_dados, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Converte timestamp (ms) para datetime legível e define como índice
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    df.drop('timestamp', axis=1, inplace=True) # Remove a coluna original

    print(f"\nBusca concluída. Total de {len(df)} registros baixados.")

    # 5. Salva o DataFrame em um arquivo CSV
    # Garante que a pasta de saída exista
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Formata o nome do arquivo
    nome_ativo_arquivo = ativo.replace('/', '_') # ex: BTC_USDT
    nome_arquivo = f"ohlcv_{nome_ativo_arquivo}_{timeframe}.csv"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    
    df.to_csv(caminho_arquivo)
    print(f"Dados salvos com sucesso em: {caminho_arquivo}")
    
    return df

def get_news_data(query, data_inicio_str, data_fim_str, pasta_saida='data/raw'):
    """
    Busca notícias de uma query usando a NewsAPI e salva em JSON.

    Parâmetros:
    - query (str): O termo de busca (ex: 'Bitcoin', 'Ethereum').
    - data_inicio_str (str): Data de início 'YYYY-MM-DD'.
    - data_fim_str (str): Data de fim 'YYYY-MM-DD'.
    - pasta_saida (str): Pasta onde o JSON será salvo.
    """
    
    # 1. Carrega variáveis de ambiente (onde está nossa chave)
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        print("Erro: Chave da NewsAPI não encontrada.")
        print("Por favor, crie um arquivo .env na raiz do projeto com 'NEWS_API_KEY=SUA_CHAVE'")
        return None

    # 2. Inicializa o cliente da API
    newsapi = NewsApiClient(api_key=api_key)

    print(f"Buscando notícias para '{query}' de {data_inicio_str} até {data_fim_str}...")

    try:
        # 3. Executa a busca
        # Nota: O plano 'developer' (gratuito) só permite buscar nos últimos 30 dias.
        # language='en' -> Focar em inglês melhora a qualidade do sentimento
        todos_os_artigos = newsapi.get_everything(
            q=query,
            language='en',
            from_param=data_inicio_str,
            to=data_fim_str,
            sort_by='publishedAt', # Ordena por data
            page_size=100 # Máximo por página
        )
        
        num_artigos = todos_os_artigos['totalResults']
        if num_artigos == 0:
            print("Nenhum artigo encontrado para esta consulta.")
            return None
            
        print(f"Total de {num_artigos} artigos encontrados.")
        
        # 4. Salva os dados brutos em um arquivo JSON
        os.makedirs(pasta_saida, exist_ok=True)
        
        # Formata o nome do arquivo
        query_arquivo = query.replace(' ', '_').lower()
        nome_arquivo = f"noticias_{query_arquivo}_{data_inicio_str}_a_{data_fim_str}.json"
        caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)

        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            json.dump(todos_os_artigos, f, ensure_ascii=False, indent=4)
            
        print(f"Notícias salvas com sucesso em: {caminho_arquivo}")
        
        return todos_os_artigos

    except Exception as e:
        print(f"Erro ao buscar notícias: {e}")
        return None