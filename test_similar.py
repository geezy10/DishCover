import requests
import json

RECIPE_ID = 1000
url = f"http://127.0.0.1:5000/recipe/{RECIPE_ID}/similar"

print(f" search similar recipes {RECIPE_ID}...")

try:
    response = requests.get(url)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print(f"\n original: '{data.get('original_title')}'")
        print("\nsimilar recipes:")

        for r in data['results']:
            print(f"- {r['title']} (Score: {r['score']:.4f})")
            print(f"  Bild: {r.get('image', 'Kein Bild')}")
            print("-" * 20)
    else:
        print(" Server Fehler:", response.text)

except Exception as e:
    print(f"Verbindungsfehler: {e}")