# 🍳 DishCover: Semantische Rezept-Empfehlung mit NLP

Ein Machine-Learning-Projekt, das Kochrezepte analysiert und ähnliche Gerichte basierend auf Zutaten-Embeddings vorschlägt. Das Modell versteht kulinarische Kontexte durch den Einsatz von **Word2Vec** und **Cosine Similarity**.
<img width="1480" height="2800" alt="Screenshot_20260123-140210-portrait" src="https://github.com/user-attachments/assets/515c32ea-d20d-47a5-8f8a-26a6f10b4223" />
<img width="1480" height="2800" alt="Screenshot_20260123-140227-portrait" src="https://github.com/user-attachments/assets/2b42bf3d-dc8a-47c2-8285-d6378e7f7a0d" />
<img width="1480" height="2800" alt="Screenshot_20260123-133518-portrait" src="https://github.com/user-attachments/assets/8ddceba8-8d5c-422a-9907-115e7b106865" />


## 🚀 Features

*   **Intelligentes Preprocessing:** Bereinigung von rohen Zutatenlisten (Entfernen von Maßeinheiten, Füllwörtern und Zubereitungsarten wie "chopped" oder "diced") unter Verwendung von NLTK.
*   **Bigramm-Erkennung:** Automatische Identifizierung von zusammengehörigen Begriffen (z. B. `olive_oil`, `peanut_butter`, `granny_smith`) mittels Gensim Phrases.
*   **Semantisches Verständnis:** Training eines Word2Vec-Modells, das versteht, dass "Limetten" und "Zitronen" ähnlicher sind als "Limetten" und "Steak".
*   **Rezept-Vektorisierung:** Berechnung eines Durchschnittsvektors für jedes Rezept, um mathematische Ähnlichkeitsberechnungen zu ermöglichen.
*   **Empfehlungssystem:** Findet Alternativen zu einem Rezept basierend auf der Kosinus-Ähnlichkeit der Vektoren.

## 🛠️ Technologie-Stack

*   **Python 3.x**
*   **Pandas:** Datenmanipulation und Management der Rezept-Datensätze.
*   **NLTK (Natural Language Toolkit):** Tokenisierung, POS-Tagging (Part-of-Speech) und Lemmatisierung.
*   **Gensim:** Training des Word2Vec-Modells und Phraser (Bigramme).
*   **Scikit-Learn:** Berechnung der Cosine Similarity Matrix.
*   **NumPy:** Vektorberechnungen.

## 📊 Die Pipeline

Das Projekt folgt einer strikten NLP-Pipeline, um aus unstrukturiertem Text nutzbare Vektoren zu machen:

1.  **Data Ingestion:** Laden des Rezept-Datensatzes (CSV).
2.  **Advanced Cleaning:**
    *   Parsing von Strings in Python-Listen.
    *   Entfernen von Zahlen, Sonderzeichen und Klammern.
    *   **POS-Tagging Filter:** Es werden primär Nomen (`NN`) und relevante Adjektive (`JJ`) behalten.
    *   **Custom Stopwords:** Aggressives Filtern von Maßeinheiten ("cup", "oz"), Verben ("chopped", "boiled") und generischen Adjektiven ("large", "fresh"), um das Rauschen zu minimieren.
3.  **Phrasing (Bigrams):** Das Modell lernt, dass `baking` und `powder` zu `baking_powder` gehören.
4.  **Embedding Training:** Ein Word2Vec-Modell (Skip-Gram) lernt 150-dimensionale Vektoren für jede Zutat.
5.  **Vectorization:** Jedes Rezept wird durch den Durchschnitt seiner Zutaten-Vektoren repräsentiert.

## 📂 Installation & Nutzung

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/dein-username/recipe-word2vec.git
    cd recipe-word2vec
    ```

2.  **Dependencies installieren:**
    ```bash
    pip install pandas numpy nltk gensim scikit-learn
    ```

3.  **NLTK Daten laden (einmalig im Script):**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

4.  **Notebook/Script ausführen:**
    Lade deinen Datensatz (`recipes.csv`) in den `data/` Ordner und starte das Training.
