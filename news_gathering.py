import requests
import pandas as pd
from google import genai
import seaborn as sns

url = "https://eventregistry.org/api/v1/article/getArticles"

params = {
    "apiKey": "API_KEY",
    "resultType": "articles",
    "articlesSortBy": "date",
    "articlesCount": 100,
    "keyword": ["Bitcoin", "Cryptocurrency"],
    "action": "getArticles"
}

response = requests.get(url, params=params)

data = response.json()
df = pd.DataFrame(data['articles']['results'])

print(df)

client = genai.Client(api_key="APIKEY")

classifications = []

for i in range(len(df)):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
                contents=(
                    f"""Classifique o sobre bitcoin em positivos, negativos ou neutros.
        Use uma escala de -1 a 1, onde -1 é muito negativo, 0 é neutro e 1 é muito positivo.
        Retorne APENAS um numero.
        Aqui está o texto do artigo: {df.iloc[i]['body']}"""
                )
    )
    classifications.append(response.text)

df['sentiment2'] = classifications

df['sentiment2'] = pd.to_numeric(df['sentiment2'], errors='coerce')

print(df['sentiment'].corr(df["sentiment2"]))
