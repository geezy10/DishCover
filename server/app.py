import os
import pickle
import time

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from nltk.stem import WordNetLemmatizer
import ast

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
CORS(app)

# server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
IMAGES_DIR = os.path.join(DATA_DIR, "images", "Food Images")
meta_path = os.path.join(MODELS_DIR, "metadata.pkl")
csv_path = os.path.join(DATA_DIR, "recipes.csv")
model_path = '../data/models/word2vec.model'
w2v_model = None
d2v_model = None
db_ingredients = None
db_recipes = None
df_recipes = None

# load metadata (photos,..)
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        df_recipes = pickle.load(f)
        print("loaded metadata(pkl)")
elif os.path.exists(csv_path):
    print("load csv")
    df_recipes = pd.read_csv(csv_path)
else:
    print("no pkl or csv")

# load w2v model
try:
    w2v_path = os.path.join(MODELS_DIR, "word2vec.model")
    if os.path.exists(w2v_path):
        w2v_model = Word2Vec.load(w2v_path)
        print("loaded word2vec model")
    else:
        print("no word2vec model")
except Exception as e:
    print(e)

# load d2v model
try:
    d2v_path = os.path.join(MODELS_DIR, "doc2vec.model")
    if os.path.exists(d2v_path):
        d2v_model = Word2Vec.load(d2v_path)
        print("loaded doc2vec model")
    else:
        print("no doc2vec model")
except Exception as e:
    print(e)

# load doc2vec vectors
try:
    doc_path = os.path.join(MODELS_DIR, "recipe_vectors.pkl")
    with open(doc_path, "rb") as f:
        db_recipes = pickle.load(f)
except Exception as e:
    print(e)

# load word2vec(ingredient) vectors
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
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        filename = filename + '.jpg'

    return request.host_url + 'images/' + filename


def calculate_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None: return 0.0

    # scalarproduct, multiply every value from vec a & b and add it up
    dot = np.dot(vec_a, vec_b)

    # length of vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # division / 0
    if norm_a == 0 or norm_b == 0: return 0.0

    # get rid of the length from the recipe by splitting to the length of the vectors, so only the angle is the endproduct
    return dot / (norm_a * norm_b)


def parse_instructions_safe(raw_instr):
    instructions_list = []
    if isinstance(raw_instr, str):
        if len(raw_instr.strip()) < 3:
            pass
        elif raw_instr.strip().startswith('['):
            try:
                instructions_list = ast.literal_eval(raw_instr)
            except:
                instructions_list = [s.strip() for s in raw_instr.split('.') if len(s) > 5]
        else:
            instructions_list = [s.strip() for s in raw_instr.split('.') if len(s) > 5]
    elif isinstance(raw_instr, list):
        instructions_list = raw_instr

    if not instructions_list or len(instructions_list) == 0:
        return ["Zubereitungsschritte laden..."]

    return instructions_list

def get_ingredient_match_info(user_ingredients, recipe_ingredients_raw):
    if pd.isna(recipe_ingredients_raw) or recipe_ingredients_raw is None:
        return [], 0, []

    user_keywords = set()
    for item in user_ingredients:
        name = item.get('name', '') if isinstance(item, dict) else str(item)
        words = name.lower().replace("_", " ").split()
        for w in words:
            if len(w) > 2:
                try:
                    user_keywords.add(lemmatizer.lemmatize(w))
                except:
                    user_keywords.add(w)

    recipe_lines = []

    if isinstance(recipe_ingredients_raw, list):
        recipe_lines = recipe_ingredients_raw

    elif isinstance(recipe_ingredients_raw, str):
        try:
            if recipe_ingredients_raw.strip().startswith("["):
                recipe_lines = ast.literal_eval(recipe_ingredients_raw)
            else:
                recipe_lines = recipe_ingredients_raw.split("\n")
        except:
            recipe_lines = recipe_ingredients_raw.replace("[", "").replace("]", "").replace("'", "").split(",")

    else:
        return [], 0, []

    recipe_lines = [str(l).strip() for l in recipe_lines if str(l).strip()]

    matched_lines = []
    for line in recipe_lines:
        clean_line = line.lower()
        for keyword in user_keywords:
            if keyword in clean_line:
                matched_lines.append(line)
                break

    matched_lines = list(set(matched_lines))
    missing_count = max(0, len(recipe_lines) - len(matched_lines))

    return matched_lines, missing_count, recipe_lines
# --------------------------- endpoints ----------------------------------#
@app.route('/status', methods=['GET'])
def status():
    count = len(df_recipes) if df_recipes is not None else 0

    return jsonify({
        'status': 'online',
        "recipe_loaded": count})


@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    data = request.get_json()
    user_ingredients = data.get('ingredients', [])

    user_filters = data.get('filters', {})
    wants_vegetarian = user_filters.get('vegetarian', False)
    wants_vegan = user_filters.get('vegan', False)
    has_nut_allergy = user_filters.get('no_nuts', False)
    print(f" 'search:' {user_ingredients} | 'filters:' {user_filters} ")

    urgent_ingredient_names = []
    for item in user_ingredients:
        if isinstance(item, dict) and float(item.get('weight', 1.0)) > 1.5:
            urgent_ingredient_names.append(item.get('name', '').lower())

    # pre-filtering
    # copy the df and check for allergy and types of food
    filtered_df = df_recipes.copy()

    if wants_vegetarian:
        filtered_df = filtered_df[filtered_df['is_vegetarian'].astype(str) == 'True']

    if wants_vegan:
        filtered_df = filtered_df[filtered_df['is_vegan'].astype(str) == 'True']

    if has_nut_allergy:
        filtered_df = filtered_df[filtered_df['has_nuts'].astype(str) == 'False']

    print(f"found {len(filtered_df)} recipes")

    valid_recipe_ids = set(filtered_df.index.tolist())

    input_vectors = []
    weights = []

    # logic for the possibilty to send objects or strings: {"name": "tomato", "weight":...} or just tomato
    for item in user_ingredients:
        if isinstance(item, dict):
            ing_name = item.get('name', '')
            weight = float(item.get('weight', 1.0))
        else:
            ing_name = str(item)
            weight = 1.0

        clean = ing_name.lower().strip().replace(" ", "_")
        clean = lemmatizer.lemmatize(clean)

        if clean in w2v_model.wv:
            input_vectors.append(w2v_model.wv[clean])
            weights.append(weight)
        elif ing_name.lower().strip() in w2v_model.wv:
            input_vectors.append(w2v_model.wv[ing_name.lower().strip()])
            weights.append(weight)
        else:
            print(f"  '{clean}' not found!")

        # cold start
        # if payload is empty
    if not input_vectors:
        sample_size = min(20, len(filtered_df))
        print("no vectors -> random selection")
        random_selection = filtered_df.sample(n=sample_size)

        response = []
        for r_id, row in random_selection.iterrows():
            raw_ingreds = row.get('Ingredients', [])
            match_list, missing, full_list = get_ingredient_match_info(user_ingredients, raw_ingreds)
            instructions = parse_instructions_safe(row.get('Instructions', ''))

            response.append({
                "id": int(r_id),
                "title": row['Title'],
                "image": get_image_url(request, row.get('Image_Name')),
                "score": 0.0,
                "ingredients_list": full_list,
                "instructions": instructions,
                "reasons": {
                    "matching_items": match_list,
                    "missing_ingredients": missing
                },
                "tags": {
                    "vegetarian": bool(row['is_vegetarian']),
                    "vegan": bool(row['is_vegan']),
                    "nuts": bool(row['has_nuts']),
                }
            })
        return jsonify({"results": response})

    # calculate the averg from all input ingredients
    query_vec = np.average(input_vectors, axis=0, weights=weights)

    # iterate over all recipes but only append that that matched the filters
    results = []
    for r_id, r_vec in db_ingredients.items():
        if r_id in valid_recipe_ids:
            score = calculate_similarity(query_vec, r_vec)
            if score > 0.4:
                results.append((r_id, score))

    # sort
    results.sort(key=lambda x: x[1], reverse=True)

    response = []
    for r_id, score in results[:20]:
        row = df_recipes.loc[r_id]
        instructions = parse_instructions_safe(row.get('Instructions', ''))

        if not instructions or len(instructions) == 0:
            continue

        match_list, missing_count, full_list = get_ingredient_match_info(user_ingredients, row.get('Ingredients', []))


        used_urgent_ingredients = [
            i for i in match_list
            if any(urgent in i.lower() for urgent in urgent_ingredient_names)
        ]

        response.append({
            "id": int(r_id),
            "title": row['Title'],
            "image": get_image_url(request, row.get('Image_Name')),
            "score": float(score),
            "ingredients_list": full_list,
            "instructions": instructions,
            "reasons": {
                "matching_items": match_list,
                "missing_count": int(missing_count),
                "used_urgent_ingredients": used_urgent_ingredients,
            },
            "tags": {"vegetarian": bool(row['is_vegetarian']), "vegan": bool(row['is_vegan']),
                     "nuts": bool(row['has_nuts'])}
        })

    duration = time.time() - start_time
    print(duration)
    return jsonify({"results": response, "debug_info": {"latency_seconds": duration}})



@app.route('/search', methods=['POST'])
def search():
    if d2v_model is None or db_recipes is None:
        return jsonify({"error": "no model loaded"}), 404

    data = request.get_json()
    query_text = data.get('query', '')
    user_ingredients = data.get('ingredients', [])

    user_filters = data.get('filters', {})
    wants_vegetarian = user_filters.get('vegetarian', False)
    wants_vegan = user_filters.get('vegan', False)
    has_nut_allergy = user_filters.get('no_nuts', False)

    filtered_df = df_recipes.copy()
    if wants_vegetarian:
        filtered_df = df_recipes[df_recipes['is_vegetarian'].astype(str) == 'True']
    if wants_vegan:
        filtered_df = df_recipes[df_recipes['is_vegan'].astype(str) == 'True']
    if has_nut_allergy:
        filtered_df = df_recipes[df_recipes['has_nuts'].astype(str) == 'False']

    valid_recipe_ids = set(filtered_df.index.tolist())

    print(f"search query: {query_text} | Filters: {user_filters}")

    tokens = query_text.lower().split()

    if len(tokens) < 5:
        print("short query detected")
        inferred_vector = d2v_model.infer_vector(tokens, epochs=200)
    else:
        inferred_vector = d2v_model.infer_vector(tokens, epochs=20)

    candidates = []

    for r_id, r_vec in db_recipes.items():
        if r_id not in valid_recipe_ids:
            continue
        score = calculate_similarity(inferred_vector, r_vec)

        if score > 0.4:
            candidates.append((r_id, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_results = candidates[:100]
    response = []
    for r_id, score in top_results:
        row = df_recipes.iloc[r_id]
        instructions = parse_instructions_safe(row.get('Instructions', ''))
        m_list, m_count, f_list = get_ingredient_match_info(user_ingredients, row.get('Ingredients', []))

        response.append({
            "id": int(r_id),
            "title": row['Title'],
            "image": get_image_url(request, row.get('Image_Name')),
            "score": float(score),
            "instructions": instructions,
            "reasons": {
                "matching_items": m_list,
                "missing_count": int(m_count),
                "is_boosted": False,
                "used_urgent_items": []
            },
            "ingredients_list": f_list,
            "tags": {"vegetarian": bool(row.get('is_vegetarian', False)), "vegan": bool(row.get('is_vegan', False))}
        })
    return jsonify({"results": response})


@app.route('/recipe/<int:recipe_id>/similar', methods=['GET'])
def similar(recipe_id):
    if db_recipes is None:
        return jsonify("error, no Doc2Vec Database loaded"), 503

    if recipe_id not in db_recipes:
        return jsonify({"error": "no such recipe"}), 404

    print(f"search similarity to recipe: {recipe_id}")

    query_vec = db_recipes[recipe_id]
    candidates = []

    for r_id, r_vec in db_recipes.items():
        if r_id == recipe_id:
            continue
        score = calculate_similarity(query_vec, r_vec)

        if score > 0.4:
            candidates.append((r_id, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_results = candidates[:10]

    response = []

    try:
        original_title = df_recipes.loc[recipe_id]['Title']
    except:
        "No Title"

    for r_id, score in top_results:
        try:
            row = df_recipes.loc[r_id]

            img_url = get_image_url(request, row.get('Image_Name'))

            response.append({
                "id": int(r_id),
                "title": row['Title'],
                "image": img_url,
                "score": float(score),
                "tags": {
                    "vegetarian": bool(row.get('is_vegetarian', False)),
                    "vegan": bool(row.get('is_vegan', False))
                }
            })
        except Exception as e:
            print(e)
            continue

    return jsonify({
        "original_id": recipe_id,
        "original_title": original_title,
        "results": response
    })


@app.route('/recipe/<int:recipe_id>', methods=['GET'])
def get_recipe_id(recipe_id):
    try:
        row = df_recipes.loc[recipe_id]
        instructions_raw = str(row['Instructions'])
        steps = [s.strip(instructions_raw) for s in instructions_raw.split('\n')
                 if len(s.strip()) > 5]

        ingredients_raw = str(row['Ingredients'])
        ingredients_list = [i.strip() for i in ingredients_raw.split('\n') if i]

        return jsonify({
            "id": int(recipe_id),
            "title": row['Title'],
            "ingredients": ingredients_list,
            "instructions": steps,
            "image": get_image_url(request, row['Image_Name']),
            "description": row.get('Description', 'Keine Beschreibung verfügbar.')
        })
    except Exception as e:
        print(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
