# Pharmaraccomender

Benvenuti nel repository di PharmaRecommender, un progetto universitario innovativo che si concentra sullo sviluppo di un algoritmo di collaborative filtering specializzato per il settore farmaceutico. 
PharmaRaccomender
PharmaRaccomender è un sistema di raccomandazione basato sui rating dei prodotti. Utilizza l'algoritmo SVD (Singular Value Decomposition) della libreria Surprise per prevedere le valutazioni dei prodotti e fornire raccomandazioni personalizzate.

# Installazione
Prima di iniziare, assicurati di avere Python installato sul tuo sistema. Inoltre, avrai bisogno di alcune librerie Python, che puoi installare eseguendo:

```pip install pandas scikit-surprise matplotlib sklearn```

# Dataset
Il sistema di raccomandazione è stato addestrato e testato utilizzando il dataset ml-25m, preso da 1000farmacie S.P.A. 


# Utilizzo
Per eseguire il sistema di raccomandazione, segui questi passi:

Assicurati di avere un file CSV contenente i rating dei prodotti. Il file dovrebbe avere almeno tre colonne: userId, productId e rating.

Modifica il percorso del file CSV nel codice per corrispondere alla posizione del tuo file:


```df = pd.read_csv(r"Percorso\al\tuo\file\ratings.csv")```

Esegui PharmaRaccomender.py per addestrare il modello e ricevere le raccomandazioni.

# Caratteristiche


Addestramento del Modello: Utilizza l'algoritmo SVD per apprendere dalle valutazioni degli utenti.

Valutazione del Modello: Calcola RMSE (Root Mean Square Error) e MAE (Mean Absolute Error) per valutare la precisione delle previsioni.

Raccomandazioni Personalizzate: Fornisce le prime N raccomandazioni per ogni utente.

Metriche Avanzate: Include il calcolo di precisione, recall, F1-score e accuratezza per valutazioni binarizzate.
Visualizzazione della Curva ROC: Mostra la curva ROC e calcola l'AUC (Area Under the Curve) per valutare la capacità del modello di distinguere tra classificazioni positive e negative.

# Esempio di Output

```
Precision: 0.76
Recall: 0.43
F1 Score: 0.55
Matrice di Confusione:
 TP: 1200    FP: 300
 FN: 450     TN: 1050
Accuracy (with threshold 1): 0.85
```

# Visualizzazione della Curva ROC
Il progetto include una visualizzazione della curva ROC per valutare la capacità del modello di distinguere tra classificazioni positive e negative dei rating. La curva ROC e l'area sotto la curva (AUC) sono metriche chiave per comprendere la performance del modello in termini di sensibilità e specificità.

![myplot](https://github.com/Pakopro01/Pharmaraccomender/assets/152997482/59a6c203-c0a2-446d-ab19-c3ca5df0423f)

# Contribuire
Se desideri contribuire al progetto, per favore considera di fare una pull request o aprire un issue per discutere ciò che vorresti cambiare.
