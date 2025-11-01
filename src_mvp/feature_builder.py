# (Imports no topo do arquivo permanecem os mesmos)
import os
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import time

def analisar_sentimento_noticias(arquivo_json_entrada, pasta_saida='data/processed'):
    """
    Lê um arquivo JSON de notícias, analisa o sentimento de cada uma usando 
    a API do Gemini e salva um DataFrame em CSV.
    """
    
    # 1. Carregar a Chave de API
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("Erro: GOOGLE_API_KEY não encontrada no arquivo .env")
        return None
        
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("API do Google Gemini configurada com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar o Gemini: {e}")
        return None

    # 2. Carregar o arquivo de notícias
    try:
        with open(arquivo_json_entrada, 'r', encoding='utf-8') as f:
            data = json.load(f)
        articles = data.get('articles', [])
        if not articles:
            print("Nenhum artigo encontrado no arquivo JSON.")
            return None
        print(f"Carregados {len(articles)} artigos de '{arquivo_json_entrada}'")
    except Exception as e:
        print(f"Erro ao ler o arquivo JSON: {e}")
        return None

    # 3. Definir o Prompt de Análise (A Mágica da "IA Quant")
    
    # --- CORREÇÃO AQUI ---
    # As chaves { e } do exemplo JSON agora estão dobradas ({{ e }})
    # para evitar o erro de .format() do Python.
    
    prompt_template = """
    Você é um analista financeiro quantitativo especializado em criptomoedas.
    Analise o sentimento do artigo de notícia abaixo, especificamente sobre o 
    impacto potencial no preço do Bitcoin (BTC).

    Considere o título e a descrição.

    Retorne sua análise APENAS no seguinte formato JSON:
    {{
      "sentiment_score": <um float de -1.0 (extremamente negativo/bearish) a +1.0 (extremamente positivo/bullish)>,
      "reasoning": "<uma breve justificativa de 1 sentença>"
    }}

    Artigo:
    ---
    Título: {titulo}
    Descrição: {descricao}
    ---
    """

    # 4. Iterar, Analisar e Coletar Resultados
    resultados = []
    
    # Limite para o MVP (ex: 50 notícias) para não estourar a API e ir rápido
    # Remova o '[:50]' para analisar tudo
    for article in articles[:50]: 
        titulo = article.get('title', '')
        descricao = article.get('description', '')

        # Pula artigos sem conteúdo
        if not descricao or not titulo or descricao == "[Removed]":
            continue

        # Formata o prompt final
        prompt_final = prompt_template.format(titulo=titulo, descricao=descricao)

        try:
            # Chama a API
            response = model.generate_content(prompt_final)
            
            # Limpa a resposta (Gemini às vezes envolve em ```json ... ```)
            resposta_texto = response.text.strip().replace("```json", "").replace("```", "")
            
            # Parseia o JSON da resposta
            analise = json.loads(resposta_texto)
            
            score = float(analise['sentiment_score'])
            reasoning = analise['reasoning']
            timestamp = article['publishedAt']
            
            resultados.append({
                'timestamp': timestamp,
                'sentiment_score': score,
                'reasoning': reasoning,
                'title': titulo
            })
            
            print(f"  > Sucesso (Score: {score:.2f}): {titulo[:50]}...")

            # IMPORTANTE: Pausa para evitar o Rate Limit da API gratuita
            time.sleep(2) # 2 segundos entre chamadas

        except json.JSONDecodeError:
            print(f"  > ERRO DE PARSING JSON: O LLM não retornou um JSON válido. Resposta: {response.text[:100]}...")
        except Exception as e:
            print(f"  > ERRO na API ou processamento: {e}")
            time.sleep(5) # Pausa maior se der erro
            
    if not resultados:
        print("Nenhum sentimento pôde ser analisado.")
        return None

    # 5. Criar DataFrame e Salvar
    df_sentimento = pd.DataFrame(resultados)
    
    # Converte o timestamp e o torna o índice (facilitará o merge depois)
    df_sentimento['datetime'] = pd.to_datetime(df_sentimento['timestamp'])
    df_sentimento = df_sentimento.set_index('datetime').sort_index()
    
    # Garante que a pasta de saída exista
    os.makedirs(pasta_saida, exist_ok=True)
    
    nome_arquivo = "sentimento_noticias.csv"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    
    df_sentimento.to_csv(caminho_arquivo)
    print(f"\nAnálise de sentimento concluída!")
    print(f"Resultados salvos em: {caminho_arquivo}")
    
    return df_sentimento

def calcular_indicadores_tecnicos(arquivo_csv_entrada, pasta_saida='data/processed'):
    """
    Lê um arquivo CSV de preços OHLCV e calcula indicadores técnicos.

    Salva um novo CSV com os indicadores em 'data/processed'.
    """
    
    # 1. Carregar o arquivo de preços
    try:
        df = pd.read_csv(arquivo_csv_entrada, index_col='datetime', parse_dates=True)
        # Assegura que o índice está ordenado
        df.sort_index(inplace=True) 
        print(f"Carregados {len(df)} registros de preços de '{arquivo_csv_entrada}'")
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV de preços: {e}")
        return None

    # 2. Calcular Indicadores (Features de Preço)
    
    # Médias Móveis (Curta e Longa)
    df['ma_curta'] = df['close'].rolling(window=10).mean() # Média de 10 dias
    df['ma_longa'] = df['close'].rolling(window=50).mean() # Média de 50 dias
    
    # RSI (Relative Strength Index)
    periodo_rsi = 14
    delta = df['close'].diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    
    media_ganho = ganho.rolling(window=periodo_rsi).mean()
    media_perda = perda.rolling(window=periodo_rsi).mean()
    
    rs = media_ganho / media_perda
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum (Variação percentual dos últimos 5 dias)
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    
    # Volatilidade (Desvio padrão dos retornos diários em 20 dias)
    df['volatilidade_20d'] = df['close'].pct_change().rolling(window=20).std()

    # 3. Remover linhas com NaN (geradas pelos indicadores iniciais)
    df_com_features = df.dropna()
    
    if df_com_features.empty:
        print("Erro: Nenhum dado restou após o cálculo dos indicadores (talvez período de window muito longo?).")
        return None
        
    print(f"Indicadores calculados. {len(df_com_features)} registros válidos restantes.")
    
    # 4. Salvar o DataFrame com features
    # (Não precisa do os.makedirs, pois a função de sentimento já deve ter criado a pasta)
    
    # Extrai o nome base do arquivo de entrada para criar um de saída
    nome_base_arquivo = os.path.basename(arquivo_csv_entrada).replace('ohlcv_', '')
    nome_arquivo_saida = f"features_precos_{nome_base_arquivo}"
    caminho_arquivo_saida = os.path.join(pasta_saida, nome_arquivo_saida)
    
    df_com_features.to_csv(caminho_arquivo_saida)
    print(f"Features de preço salvas com sucesso em: {caminho_arquivo_saida}")
    
    return df_com_features

def criar_dataset_final(arquivo_features_precos, arquivo_sentimento, pasta_saida='data/processed'):
    """
    Junta os dados de preço (com indicadores) e os dados de sentimento
    para criar o dataset final de treinamento.
    
    Também cria a coluna 'alvo' (target) para a previsão.
    """
    
    # 1. Carregar os dois DataFrames
    try:
        df_precos = pd.read_csv(arquivo_features_precos, index_col='datetime', parse_dates=True)
        df_sentimento = pd.read_csv(arquivo_sentimento, index_col='datetime', parse_dates=True)
        
        # --- CORREÇÃO AQUI ---
        # O CSV de preços é 'naive'. Precisamos 'localizá-lo' para UTC.
        # (Sabemos que os dados da Binance/ccxt são em UTC)
        if df_precos.index.tz is None:
            df_precos.index = df_precos.index.tz_localize('UTC')
        
        # O CSV de sentimento já deve ser 'aware' (pois veio da API).
        # Apenas por garantia, convertemos para UTC caso não esteja.
        df_sentimento.index = df_sentimento.index.tz_convert('UTC')
        
        print("Arquivos de features de preço e sentimento carregados e padronizados para UTC.")
        
    except Exception as e:
        print(f"Erro ao carregar arquivos: {e}")
        return None

    # 2. Processar Sentimento (Agregar por Dia)
    # resample('D') agrupa por dia (UTC). .mean() calcula a média.
    df_sentimento_diario = df_sentimento['sentiment_score'].resample('D').mean()
    
    # Preenche dias sem notícias (NaN) com 0.0 (sentimento neutro)
    df_sentimento_diario = df_sentimento_diario.fillna(0.0)
    
    # Renomeia a coluna para o merge
    df_sentimento_diario = df_sentimento_diario.to_frame(name='sentimento_medio')
    
    print("Sentimento agregado por dia (média diária).")

    # 3. Juntar (Merge) os DataFrames
    # Agora ambos são 'tz-aware' em UTC, o join funcionará.
    df_final = df_precos.join(df_sentimento_diario, how='left')
    
    # O sentimento (últimos 30 dias) não cobrirá todo o histórico de preço (desde 2018).
    # Preenchemos o sentimento antigo (NaN) com 0.0 (neutro).
    df_final['sentimento_medio'] = df_final['sentimento_medio'].fillna(0.0)

    # 4. Criar a Coluna Alvo (Target)
    periodo_previsao = 1 # 1 dia no futuro
    df_final['target_preco_futuro'] = df_final['close'].shift(-periodo_previsao)
    df_final['target_preco_subiu'] = (df_final['target_preco_futuro'] > df_final['close']).astype(int)
    
    # 5. Limpeza Final
    df_final = df_final.dropna()
    df_final = df_final.drop(columns=['target_preco_futuro'])
    
    print("Dataset final unido e coluna 'target' criada.")
    
    # 6. Salvar em Parquet
    nome_arquivo_saida = "dataset_final_treinamento.parquet"
    caminho_arquivo_saida = os.path.join(pasta_saida, nome_arquivo_saida)
    
    df_final.to_parquet(caminho_arquivo_saida)
    
    print(f"Dataset final salvo com sucesso em: {caminho_arquivo_saida}")
    
    return df_final