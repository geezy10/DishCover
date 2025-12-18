import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS



app =  Flask(__name__)
CORS(app)

#server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
IMAGES_DIR = os.path.join(DATA_DIR, "images", "Food Images")
meta_path = os.path.join(MODELS_DIR, "metadata.pkl")
csv_path = os.path.join(DATA_DIR, "recipes.csv")
w2v_model = None
db_ingredients = None
db_recipes = None
df_recipes = None

#load metadata (photos,..)
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        df_recipes = pickle.load(f)
        print("loaded metadata(pkl)")
elif os.path.exists(csv_path):
    print("load csv")
    df_recipes = pd.read_csv(csv_path)
else:
    print("no pkl or csv")

#load w2v model
try:
    w2v_model = os.path.join(MODELS_DIR, "word2vec.model")
except Exception as e:
    print(e)

#load doc2vec model
try:
    doc_path = os.path.join(MODELS_DIR, "recipe_vectors.pkl")
    with open(doc_path, "rb") as f:
        db_recipes = pickle.load(f)
except Exception as e:
    print(e)

#load ingredient vectors
try:
    ing_path = os.path.join(MODELS_DIR, "ingredient_vectors.pkl")
    if os.path.exists(ing_path):
        with open(ing_path, "rb") as f:
            db_ingredients = pickle.load(f)
            print("loaded ingredients(pkl)")
    else:
        print("no pkl")
except Exception as e:
    print(e)


def get_image_url(request, image_name):
    if pd.isna(image_name) or str(image_name) == 'nan':
        return None

    filename = str(image_name)
    if not filename.lower().endswith(('.jpg','png','jpeg')):
        filename = filename + '.jpg'

    return request.host_url + 'images/' + filename


#--------------------------- endpoints ----------------------------------#
@app.route('/status', methods=['GET'])
def status():
    count = len(df_recipes) if df_recipes is not None else 0

    return jsonify({
        'status': 'online',
        "recipe_loaded": count})


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)