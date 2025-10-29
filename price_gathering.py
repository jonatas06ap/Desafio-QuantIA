# import ccxt
# import pandas as pd
# from datetime import datetime
#
# # 1. Inicializa o conector da exchange (sem necessidade de chaves de API para dados públicos)
# exchange = ccxt.binance({
#     'rateLimit': 1200,  # Respeita o limite da API
#     'enableRateLimit': True
# })
#
# def fetch_historical_ohlcv(symbol, timeframe, start_date_str):
#     """
#     Busca dados OHLCV históricos de uma exchange.
#     
#     :param symbol: O par, ex: 'BTC/USDT'
#     :param timeframe: A granularidade, ex: '1h', '4h', '1d'
#     :param start_date_str: Data de início no formato ISO 8601, ex: '2023-01-01T00:00:00Z'
#     """
#     print(f"Iniciando busca por {symbol} ({timeframe}) a partir de {start_date_str}...")
#     
#     # ccxt usa timestamps em milissegundos
#     since_timestamp = exchange.parse8601(start_date_str)
#     
#     all_ohlcv = []
#     
#     # Loop para "paginar" os resultados (backfill)
#     # A Binance limita a 1000 velas por chamada
#     while True:
#         try:
#             # Busca os dados
#             ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)
#             
#             if len(ohlcv) == 0:
#                 break # Saímos do loop se não houver mais dados
#
#             # O primeiro registro é o mais antigo
#             all_ohlcv.extend(ohlcv)
#             
#             # Define o 'since' da próxima chamada para o timestamp da *última* vela + 1
#             since_timestamp = ohlcv[-1][0] + 1 
#             
#             print(f"Recebidas {len(ohlcv)} velas. Último timestamp: {exchange.iso8601(ohlcv[-1][0])}")
#
#         except ccxt.NetworkError as e:
#             print(f"Erro de rede: {e}. Tentando novamente...")
#             exchange.sleep(5000) # Espera 5s
#         except Exception as e:
#             print(f"Erro: {e}")
#             break
#             
#     if len(all_ohlcv) == 0:
#         return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#
#     # 2. Converte a lista de listas em um DataFrame do Pandas
#     df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     
#     # 3. Limpeza e Formatação (Crucial para a Fase 3)
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     df.set_index('timestamp', inplace=True)
#     df = df.astype(float) # Garante que todos os dados são numéricos
#     
#     # Remove duplicados (caso a API retorne sobreposições)
#     df = df[~df.index.duplicated(keep='first')]
#     
#     print(f"Busca concluída. Total de {len(df)} velas.")
#     return df
#
# # --- Exemplo de Uso (Backfill) ---
# # (Isto pode demorar alguns minutos para rodar pela primeira vez)
# # df_btc_1h = fetch_historical_ohlcv('BTC/USDT', '1h', '2020-01-01T00:00:00Z')
# # print(df_btc_1h.head())
# # print(df_btc_1h.tail())
#
# # --- Exemplo de Uso (Atualização Contínua) ---
# # Para o pipeline agendado, você buscaria apenas as últimas 24h
# df_btc_update = fetch_historical_ohlcv('BTC/USDT', '1h', '2025-10-27T00:00:00Z') # Busca dos últimos 2 dias
# print(df_btc_update)