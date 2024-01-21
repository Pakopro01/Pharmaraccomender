import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise.accuracy import rmse

# Carica i dati dal file CSV in un DataFrame
df = pd.read_csv(r"C:\Users\Paolo Colantuoni\OneDrive - 1000farmacie SPA\Desktop\DATA MINING\dataset_fittizio_raccomandazione.csv")

# Crea il 'Reader' specificando la scala delle valutazioni (se necessario)
reader = Reader(rating_scale=(1, 5))

# Carica il dataset di valutazioni in 'surprise' dal DataFrame
data = Dataset.load_from_df(df[['ID Utente', 'ID Prodotto', 'Valutazione']], reader)

from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise.accuracy import rmse

# Carica i dati nel formato richiesto da 'surprise'
reader = Reader(rating_scale=(1, 5))


# Suddividi i dati in set di addestramento e test
trainset, testset = train_test_split(data, test_size=0.2)
print(testset)

# Utilizza il KNNBasic come algoritmo di filtraggio collaborativo
algo = KNNBasic()

# Addestra il modello sul set di addestramento
algo.fit(trainset)

# Fai previsioni sul set di test
predictions = algo.test(testset)

# Calcola l'RMSE
accuracy = rmse(predictions)

from surprise import accuracy

# Previsioni sul set di test
predictions = algo.test(testset)

# Calcolo del MAE
mae_value = accuracy.mae(predictions)
from collections import defaultdict

def get_top_n(predictions, n=10):
    '''Restituisce le prime N raccomandazioni per ciascun utente dal set di previsioni.'''

    # Mappa le previsioni per ciascun utente
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Ordina le previsioni per ciascun utente e seleziona le N più alte
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Soglia per considerare una previsione come 'positiva'
threshold = 4.0

# Previsioni sul set di test
predictions = algo.test(testset)
print(predictions)
input()
# Calcolo delle top-N raccomandazioni per ciascun utente
top_n = get_top_n(predictions, n=10)

# Calcolo di precisione e recall
def calculate_precision_recall(top_n, threshold):
    hits = 0
    total_pred = 0
    total_rel = 0

    for uid, user_ratings in top_n.items():
        # Previsioni rilevanti (sopra soglia)
        rel_pred = [iid for (iid, est) in user_ratings if est >= threshold]
        # Verifica quanti sono veramente rilevanti
        hits += sum((true_r >= threshold) for (iid, true_r) in user_ratings)
        # Conta totali
        total_pred += len(rel_pred)
        total_rel += sum((true_r >= threshold) for (_, true_r) in user_ratings)

    precision = hits / total_pred if total_pred != 0 else 0
    recall = hits / total_rel if total_rel != 0 else 0
    return precision, recall

precision, recall = calculate_precision_recall(top_n, threshold)

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')
