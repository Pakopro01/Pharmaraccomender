import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# Carica i dati dal file CSV in un DataFrame
df = pd.read_csv(r"C:\Users\Paolo Colantuoni\Downloads\ml-25m\ml-25m\ratings.csv")

# Crea il 'Reader' specificando la scala delle valutazioni (se necessario)
reader = Reader(rating_scale=(1, 5))

# Carica il dataset di valutazioni in 'surprise' dal DataFrame
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Suddividi i dati in set di addestramento e test
trainset, testset = train_test_split(data, test_size=0.2)

# Utilizza l'SVD come algoritmo di filtraggio collaborativo
algo = SVD()

# Addestra il modello sul set di addestramento
algo.fit(trainset)

# Fai previsioni sul set di test
predictions = algo.test(testset)

# Calcola l'RMSE e il MAE
accuracy_rmse = rmse(predictions)
accuracy_mae = mae(predictions)

from collections import defaultdict


def get_top_n(predictions, n=10):
    '''Restituisce le prime N raccomandazioni per ciascun utente dal set di previsioni.'''
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Restituisce la precisione e il recall al cutoff k per ciascun utente.'''
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Numero di raccomandazioni rilevanti (true positives)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings[:k])

        # Numero di raccomandazioni rilevanti e raccomandate (true positives)
        n_rel_and_rec_k = sum((est >= threshold) and (true_r >= threshold) for (est, true_r) in user_ratings[:k])

        # Numero di raccomandazioni raccomandate (true positives + false positives)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Precisione e recall
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def f1_score(precision, recall):
    '''Calcola lo F1-score date precisione e recall.'''
    if (precision + recall) != 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0
# Assumi che `predictions` sia il risultato di algo.test(testset)
top_n = get_top_n(predictions, n=10)
precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3.0)

# Calcola la media della precisione e del recall su tutti gli utenti
average_precision = sum(prec for prec in precisions.values()) / len(precisions)
average_recall = sum(rec for rec in recalls.values()) / len(recalls)
average_f1 = f1_score(average_precision, average_recall)

print(f'Precision: {average_precision:.4f}')
print(f'Recall: {average_recall:.4f}')
print(f'F1 Score: {average_f1:.4f}')

threshold = 4.0  # Soglia per definire un rating come positivo
tp = 0  # Veri Positivi
fp = 0  # Falsi Positivi
tn = 0  # Veri Negativi
fn = 0  # Falsi Negativi

# Assumendo che `predictions` sia il risultato di algo.test(testset)
for pred in predictions:
    actual = pred.r_ui  # Rating reale
    est = pred.est  # Rating stimato

    if actual >= threshold:
        if est >= threshold:
            tp += 1
        else:
            fn += 1
    else:
        if est >= threshold:
            fp += 1
        else:
            tn += 1

# Costruzione della matrice di confusione
confusion_matrix = [[tp, fp], [fn, tn]]

# Visualizzazione della matrice di confusione
print("Matrice di Confusione:")
print(f" TP: {tp}\tFP: {fp}")
print(f" FN: {fn}\tTN: {tn}")

def calculate_accuracy(predictions, threshold=1):
    """Calcola l'accuratezza come la percentuale di previsioni entro una certa soglia dal rating vero."""
    correct_predictions = sum(1 for pred in predictions if abs(pred.est - pred.r_ui) < threshold)
    return correct_predictions / len(predictions) if predictions else 0

# Dopo aver eseguito le previsioni con `predictions = algo.test(testset)`
accuracy = calculate_accuracy(predictions, threshold=1)

print(f'Accuracy (with threshold 1): {accuracy:.4f}')

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assumendo che 'predictions' sia il tuo set di previsioni
actuals = np.array([pred.r_ui for pred in predictions])
estimates = np.array([pred.est for pred in predictions])

# Converti i rating in classi binarie basate su una soglia, es. 4.0
threshold = 4.0
actual_classes = (actuals >= threshold).astype(int)

# Calcola i valori per la curva ROC
fpr, tpr, thresholds = roc_curve(actual_classes, estimates)
roc_auc = auc(fpr, tpr)

# Visualizzazione della curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

