import ccxt
import pandas as pd
import os
from datetime import datetime
import time
import json
from newsapi import NewsApiClient
from dotenv import load_dotenv
import requests

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


# As datas de entrada devem ser strings no formato "YYYY-MM-DD"
def get_news_data(data_inicio_str, data_fim_str, pasta_saida='data/raw'):

    # 1. Carrega variáveis de ambiente (onde está nossa chave)
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        print("Erro: Chave da NewsAPI não encontrada.")
        print("Por favor, crie um arquivo .env na raiz do projeto com 'NEWS_API_KEY=SUA_CHAVE'")
        return None

    # 2. Define os parâmetros da requisição
    url_api = "https://eventregistry.org/api/v1/article/getArticles"
    params =  {
        "action": "getArticles",
        "keyword": ["Bitcoin", "Cryptocurrency", "Blockchain", "Crypto"],
        "startDate": data_inicio_str,
        "endDate": data_fim_str,
        "ignoreSourceGroupUri": "paywall/paywalled_sources",
        # "articlesPage": 1, # Será definido dentro do loop
        "articlesSortBy": "date",
        "dataType": [
            "news",
            "pr"
        ],
        "resultType": "articles",
        "apiKey": api_key,
        "articlesCount": 100 # Explícito que queremos 100 por página
    }

    print(f"Buscando notícias de {data_inicio_str} até {data_fim_str}...")

    # --- Início das Alterações ---

    pagina_atual = 1
    todos_os_artigos = [] # Lista para acumular todos os resultados

    # 3. Loop de paginação
    while True:
        try:
            # Define a página atual nos parâmetros da requisição
            params["articlesPage"] = pagina_atual
            
            print(f"Buscando página {pagina_atual}...")
            response = requests.get(url_api, params=params)
            response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins
            dados = response.json()

            # Extrai os artigos desta página
            # Usamos .get() para evitar KeyErrors se a resposta for inesperada
            artigos_da_pagina = dados.get('articles', {}).get('results', [])
            
            if not artigos_da_pagina:
                # Se 'results' está vazio, não há mais páginas.
                print("Não há mais artigos. Finalizando a busca.")
                break # Sai do loop 'while True'

            # Adiciona os artigos encontrados à lista principal
            todos_os_artigos.extend(artigos_da_pagina)
            
            # Feedback opcional sobre o total de páginas
            if pagina_atual == 1: # Só executa na primeira página
                total_paginas = dados.get('articles', {}).get('pages', 0)
                if total_paginas > 0:
                    print(f"Total de páginas a serem buscadas: {total_paginas}")

            # Prepara para a próxima iteração
            pagina_atual += 1
            
            # Adiciona um pequeno delay para ser gentil com a API
            time.sleep(0.5) # 0.5 segundos

        except requests.exceptions.RequestException as e:
            print(f"Erro ao buscar notícias na página {pagina_atual}: {e}")
            print("Parando a coleta para evitar dados incompletos.")
            return None # Retorna None em caso de falha
            
    # --- Fim das Alterações ---

    # Se chegamos aqui, o loop terminou com sucesso
    num_artigos_total = len(todos_os_artigos)

    if num_artigos_total == 0:
        print("Nenhum artigo encontrado no período especificado.")
        return None

    print(f"Total de {num_artigos_total} artigos encontrados em {pagina_atual - 1} páginas.")

    # 4. Salva os dados *completos* em um arquivo JSON
    os.makedirs(pasta_saida, exist_ok=True)
    
    nome_arquivo = f"noticias_eventregistry_{data_inicio_str}_a_{data_fim_str}.json"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)

    # Recria a estrutura de resposta da API, mas com os resultados agregados
    dados_completos = {
        "articles": {
            "results": todos_os_artigos,
            "totalResults": num_artigos_total,
            "pages": pagina_atual - 1
        }
    }

    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados_completos, f, ensure_ascii=False, indent=4)
        
    print(f"Notícias salvas com sucesso em: {caminho_arquivo}")
    
    # Retorna o dicionário completo com todos os artigos
    return dados_completos