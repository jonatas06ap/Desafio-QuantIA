 # Em src_mvp/model_trainer.py

"""
Módulo de Treinamento de Modelo (Fase 4)

Este script carrega o dataset de features, seleciona as features técnicas
(ignorando o sentimento, conforme a estratégia), divide os dados cronologicamente
e treina um modelo de classificação (XGBoost).

Salva dois artefatos:
1. O modelo treinado (ex: 'model.pkl').
2. Um JSON com as métricas de avaliação (ex: 'metrics.json').
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # Para salvar o modelo de forma eficiente
import json
import argparse
import sys
import os

def split_data_chronological(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Divide os dados de séries temporais em treino e teste cronologicamente.
    NÃO usa 'train_test_split' aleatório para evitar data leakage.
    """
    print(f"Dividindo dados cronologicamente (Test size: {test_size})...")
    
    # Calcula o índice de divisão
    split_index = int(len(X) * (1 - test_size))
    
    # Divide os dataframes
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"  {len(X_train)} amostras de treino (de {X_train.index.min()} a {X_train.index.max()})")
    print(f"  {len(X_test)} amostras de teste (de {X_test.index.min()} a {X_test.index.max()})")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Treina um modelo XGBoost Classifier.
    """
    print("Iniciando treinamento do XGBClassifier...")
    
    # Define o modelo
    # 'use_label_encoder=False' e 'eval_metric='logloss'' são boas práticas
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Treina o modelo
    model.fit(X_train, y_train)
    
    print("Treinamento concluído.")
    return model

def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Avalia o modelo no conjunto de teste e retorna um dicionário de métricas.
    """
    print("Avaliando modelo no conjunto de teste...")
    
    # Faz as predições
    y_pred = model.predict(X_test)
    
    # Calcula métricas
    accuracy = accuracy_score(y_test, y_pred)
    # Gera o relatório de classificação (precisão, recall, f1)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    # Gera a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Relatório de Avaliação ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    print("Matriz de Confusão:")
    print(cm)
    print("------------------------------")
    
    # Prepara métricas para salvar em JSON
    metrics = {
        'accuracy': accuracy,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist() # Converte array numpy para lista
    }
    
    return metrics

def run_training_pipeline(data_path: str, model_output_path: str, metrics_output_path: str):
    """
    Orquestra o pipeline de treinamento completo.
    """
    print(f"Iniciando pipeline de treinamento com dados de: {data_path}")
    
    # 1. Carregar Dados
    try:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Erro: Arquivo de dados não encontrado em {data_path}", file=sys.stderr)
        return
    
    # 2. Definir Features (X) e Alvo (y)
    # --- ESTE É O PASSO CRÍTICO ---
    # Ignoramos explicitamente as colunas de sentimento
    cols_to_ignore = ['target', 'sentiment_mean', 'news_volume']
    features = [col for col in df.columns if col not in cols_to_ignore]
    target = 'target'
    
    print(f"Treinando com {len(features)} features (sentimento IGNORADO):")
    print(features)
    
    X = df[features]
    y = df[target]
    
    # 3. Dividir Dados
    # Usamos o split cronológico
    X_train, X_test, y_train, y_test = split_data_chronological(X, y, test_size=0.2) # 80% treino, 20% teste
    
    # 4. Treinar Modelo
    model = train_model(X_train, y_train)
    
    # 5. Avaliar Modelo
    metrics = evaluate_model(model, X_test, y_test)
    
    # 6. Salvar Artefatos (Modelo e Métricas)
    print(f"Salvando modelo em: {model_output_path}")
    joblib.dump(model, model_output_path)
    
    print(f"Salvando métricas em: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Pipeline de treinamento concluído com sucesso.")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento do Modelo (Fase 4)")
    
    parser.add_argument(
        "data_file", 
        help="Caminho para o 'dataset_treino_final.csv' (saída da Fase 3)."
    )
    parser.add_argument(
        "--model_out", 
        default="trained_model.pkl", 
        help="Caminho para salvar o artefato do modelo treinado."
    )
    parser.add_argument(
        "--metrics_out", 
        default="training_metrics.json", 
        help="Caminho para salvar o JSON com as métricas de avaliação."
    )
    
    args = parser.parse_args()
    
    # Assegura que os diretórios de saída existam
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    
    run_training_pipeline(args.data_file, args.model_out, args.metrics_out)
