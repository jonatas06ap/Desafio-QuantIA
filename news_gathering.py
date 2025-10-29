import requests
import pandas as pd
from datetime import datetime, timedelta

# ATENÇÃO: Substitua pela sua chave obtida no newsapi.org
NEWS_API_KEY = "SUA_CHAVE_API_AQUI" 

def fetch_crypto_news(keywords, from_datetime):
    """
    Busca notícias usando a NewsAPI.
    
    :param keywords: String de busca, ex: "bitcoin OR ethereum"
    :param from_datetime: Objeto datetime de quando começar a busca
    """
    if NEWS_API_KEY == "SUA_CHAVE_API_AQUI":
        print("Erro: Por favor, configure sua NEWS_API_KEY.")
        return pd.DataFrame()
        
    base_url = "https://newsapi.org/v2/everything"
    
    # Formata a data para a API
    from_iso = from_datetime.isoformat()
    
    params = {
        'q': keywords,
        'from': from_iso,
        'sortBy': 'publishedAt', # Garante a ordem cronológica
        'language': 'en', # 'pt' também é uma opção
        'apiKey': NEWS_API_KEY,
        'pageSize': 100 # Máximo por chamada
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Lança um erro se a requisição falhar
        
        data = response.json()
        articles = data.get('articles', [])
        
        if not articles:
            print("Nenhum artigo novo encontrado.")
            return pd.DataFrame()
            
        print(f"Encontrados {len(articles)} artigos.")
        
        # 1. Filtra os dados que queremos (como definido na Fase 1)
        cleaned_articles = []
        for article in articles:
            cleaned_articles.append({
                'published_at': article['publishedAt'],
                'source_name': article['source']['name'],
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'raw_text': article['content'] # O texto bruto para o LLM
            })
            
        # 2. Converte para DataFrame
        df = pd.DataFrame(cleaned_articles)
        
        # 3. Limpeza e Formatação
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP: {http_err} - Verifique sua API Key ou parâmetros.")
    except Exception as e:
        print(f"Erro ao buscar notícias: {e}")
        
    return pd.DataFrame()

# --- Exemplo de Uso (Pipeline Contínuo) ---
# Buscar notícias das últimas 12 horas
agora = datetime.utcnow()
inicio_busca = agora - timedelta(hours=12)

df_noticias = fetch_crypto_news(
    keywords="(bitcoin OR btc OR ethereum OR eth) AND (bull OR bear OR regulation OR adoption)",
    from_datetime=inicio_busca
)

print(df_noticias.head())