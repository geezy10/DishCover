# 🍳 FeedMe: ML-Backend & Recommendation API

Dieses Repository enthält das Python-Backend und die Machine-Learning-Logik für **FeedMe**, ein intelligentes Empfehlungssystem zur nachhaltigen Lebensmittelverwertung. Das System analysiert den digitalen Haushaltsbestand des Nutzers und schlägt basierend auf **Word2Vec-Embeddings** und **hybriden Filteralgorithmen** passende Rezepte vor, um Lebensmittelverschwendung (Food Waste) proaktiv zu reduzieren.

<p align="center">
  <img width="32%" src="https://github.com/user-attachments/assets/4397cf03-65ff-4567-b261-eace166f06b9" alt="FeedMe Übersicht" />
  <img width="32%" src="https://github.com/user-attachments/assets/060cb055-e5cf-42e4-ade8-8ba0b09581e7" alt="FeedMe Detailansicht" />
  <img width="32%" src="https://github.com/user-attachments/assets/0a806515-1d74-480a-90d1-c46bd8cb44d1" alt="FeedMe Kochmodus" />
</p>
## 🚀 Features & Innovation



*   **Proaktives MHD-Boosting:** Zutaten, deren Mindesthaltbarkeitsdatum (MHD) bald abläuft, erhalten bei der Vektorberechnung ein mathematisch höheres Gewicht. Dadurch rutschen Rezepte zur Resteverwertung automatisch im Ranking nach oben.
*   **Hybrides Empfehlungssystem:** Kombination aus deterministischem Pre-Filtering (z.B. strikte Einhaltung von "Vegan" oder "Nussfrei") und semantischer Ähnlichkeitsberechnung (Cosine Similarity).
*   **Dynamische Semantische Suche:** Eine Freitextsuche, die das aktuelle Inventar des Nutzers berücksichtigt und die Suchergebnisse durch ein dynamisches Re-Ranking anpasst.
*   **Intelligentes NLP-Preprocessing:** Bereinigung von rohen Zutatenlisten (Entfernen von Maßeinheiten und Füllwörtern) mittels `NLTK` sowie Erkennung semantischer Einheiten (Bigramme wie `olive_oil`) mittels `Gensim Phrases`.
*   **Explainable AI (XAI):** Die API liefert detaillierte Metadaten zurück (z.B. `used_urgent_ingredients`, `missing_count`), um die Entscheidungen des Algorithmus für das Frontend transparent und nachvollziehbar zu machen.

## 🛠️ Technologie-Stack

*   **Backend Framework:** Python 3.x, Flask (REST-API)
*   **Machine Learning:** Gensim (Word2Vec Skip-Gram), Scikit-Learn (Cosine Similarity Matrix)
*   **Data Science:** Pandas, NumPy
*   **NLP:** NLTK (Tokenisierung, POS-Tagging, Lemmatisierung)

## 📊 Die ML-Pipeline

Das Projekt folgt einer strikten Pipeline, um aus unstrukturiertem Text nutzbare Vektoren für das Matching zu generieren:

1.  **Data Cleaning:** Parsing von Strings, Entfernen von Zahlen/Sonderzeichen. NLTK POS-Tagging behält primär Nomen und relevante Adjektive, während Custom-Blacklists Maßeinheiten und Verben verwerfen.
2.  **Phrasing (Bigrams):** Das Modell lernt, dass Zusammenhänge wie `baking` + `powder` zu `baking_powder` verschmelzen.
3.  **Embedding Training:** Ein Word2Vec-Modell (Skip-Gram) lernt dichte semantische Vektoren für jede Zutat.
4.  **Vectorization:** Jedes Rezept wird durch den aggregierten Durchschnittsvektor seiner Zutaten repräsentiert.

## 📂 Installation & Nutzung

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/DEIN-USERNAME/feedme-backend.git
    cd feedme-backend
    ```

2.  **Virtual Environment erstellen & Dependencies installieren:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **NLTK Daten laden (einmalig):**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

4.  **Server starten:**
    Stelle sicher, dass die Trainingsdaten (`.pkl` / `.csv`) im vorgesehenen Verzeichnis liegen.
    ```bash
    python app.py
    # Der Flask-Server läuft nun standardmäßig auf http://localhost:5000
    ```

## 📡 API Endpunkte (Auszug)

*   `POST /recommend` – Generiert Rezeptvorschläge basierend auf dem übergebenen Inventar (inkl. MHD-Gewichten) und aktiven Filtern.
*   `POST /search` – Semantische Freitextsuche mit optionalem Inventar-Re-Ranking.
*   `GET /recipe/<id>` – Ruft detaillierte Rezeptinformationen ab.
