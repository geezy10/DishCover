import requests

url = "http://127.0.0.1:5000/recommend"

# payload = {}
#payload = {"ingredients": [ "delicious"]}

# payload = {
#
#     "ingredients": ["mozzarella", "tomato", "basil", "Olive Oil"],
#     "filters": {
#         "vegetarian": true,
#         "no_nuts": True,
#         "vegan": True,
#     }
# }


payload = {
    "ingredients": [
        {"name": "mozzarella"},
        {"name": "tomato"},
        "basil",
        "zucchini",
        "chicken breast"

    ],
    "filters": {"vegetarian": False,
                "no_nuts": False,
                "vegan": False,
                }

}

print(f"📡 send request to {url}...")
print(f" payload: {payload}")

try:
    response = requests.post(url, json=payload)

    print(f"status code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()

        print(f"\n raw data: {len(data.get('results', []))}")

        print("\n Recommendations: ")
        for recipe in data.get('results', []):
            title = recipe.get('title', 'no title')
            score = recipe.get('score', 0.0)
            image = recipe.get('image', 'no image')

            print(f"- {title} (Score: {score:.2f})")
            print(f"  image: {image}")
            print("-" * 20)
    else:
        print(" Error:", response.text)

except Exception as e:
    print(f"Error: {e}")
