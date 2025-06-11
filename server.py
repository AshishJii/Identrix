import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import face_recognition
import tempfile
import shutil
import logging
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MONGO_URI = os.getenv('MONGO_URI')
MONGO_DATABASE_NAME = os.getenv('MONGO_DATABASE_NAME')
MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

mongo_client = None
mongo_db = None
mongo_collection = None
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DATABASE_NAME]
    mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
    logging.info(f"MongoDB Atlas connected successfully to database '{MONGO_DATABASE_NAME}' and collection '{MONGO_COLLECTION_NAME}'.")
except Exception as e:
    logging.error(f"Error connecting to MongoDB Atlas: {e}")

app = Flask(__name__)

def downscale(path, max_dim=800):
    img = Image.open(path)
    # choose the right resampling constant for your Pillow version
    try:
        resample = Image.Resampling.LANCZOS    # Pillow ≥ 9.1.0
    except AttributeError:
        resample = Image.LANCZOS               # older Pillow

    img.thumbnail((max_dim, max_dim), resample=resample)
    img.save(path)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_face_embedding(image_path):
    print(2.1)
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None
    if os.path.splitext(image_path)[1][1:].lower() not in ALLOWED_EXTENSIONS:
        logging.error(f"Unsupported image format for {image_path}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        return None

    print(2.2)
    try:
        image = face_recognition.load_image_file(image_path)
        print(2.3)
        face_bounding_boxes = face_recognition.face_locations(image)
        print(2.4)

        if len(face_bounding_boxes) == 0:
            logging.warning(f"No face found in image: {image_path}. Cannot generate embedding.")
            return None
        elif len(face_bounding_boxes) > 1:
            logging.warning(f"Multiple faces found in image: {image_path}. Cannot reliably choose one. Skipping embedding.")
            return None
        else:
            face_encoding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
            return face_encoding.tolist()
    except Exception as e:
        logging.error(f"Error processing image {image_path} for embedding: {e}")
        return None

def search_mongodb_for_face(collection, query_embedding, limit=5):
    if not query_embedding:
        logging.error("No query embedding provided for vector search.")
        return []
    logging.info(f"Searching MongoDB collection '{collection.name}' using $vectorSearch...")

    numCandidates = None
    if limit<10:
        numCandidates = 50
    else:
        numCandidates = limit*5
    try:
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'face',
                    'path': 'faceEmbedding',
                    'queryVector': query_embedding,
                    'numCandidates': numCandidates,
                    'limit': limit
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'documentId': {'$toString': '$_id'},
                    'name': '$name',
                    'ID': '$ID',
                    'rollNo': '$rollNo',
                    'email': '$email',
                    'address': '$address',
                    'hostel': '$hostel',
                    'fatherName': '$fatherName',
                    'mobile': '$mobile',
                    'section': '$section',
                    'dob': '$dob',
                    'distance': { '$meta': 'vectorSearchScore' }
                }
            }
        ]

        results = list(collection.aggregate(pipeline))

        logging.info(f"$vectorSearch returning {len(results)} matches.")
        return results

    except PyMongoError as e:
        logging.error(f"Error during MongoDB $vectorSearch: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during MongoDB vector search: {e}")
        return []

# --- API Endpoint 1: Text-based Search ---
@app.route('/search', methods=['POST'])
def search_documents():
    if mongo_client is None:
        return jsonify({"error": "Database connection not established."}), 500

    try:
        query_params = request.get_json()

        if not query_params:
            return jsonify({"error": "No query parameters provided."}), 400

        mongo_filter = {}
        limit_value = None

        searchable_fields = {
            "ID": {"type": int, "mongo_field": "ID"},
            "rollNo": {"type": int, "mongo_field": "rollNo"},
            "name": {"type": str, "mongo_field": "name"},
            "address": {"type": str, "mongo_field": "address"},
            "email": {"type": str, "mongo_field": "email"},
            "hostel": {"type": bool, "mongo_field": "hostel"},
            "fatherName": {"type": str, "mongo_field": "fatherName"},
            "mobile": {"type": str, "mongo_field": "mobile"},
            "section": {"type": str, "mongo_field": "section"},
            "dob": {"type": str, "mongo_field": "dob"}
        }

        for param, value in query_params.items():
            if param == 'limit':
                limit_value = value
                continue

            if param in searchable_fields:
                field_info = searchable_fields[param]
                mongo_field_name = field_info["mongo_field"]
                expected_type = field_info["type"]

                if not isinstance(value, expected_type):
                    return jsonify({
                        "error": f"Invalid type for '{param}'. Expected {expected_type.__name__}, got {type(value).__name__}."
                    }), 400

                if expected_type == str:
                    mongo_filter[mongo_field_name] = {"$regex": str(value), "$options": "i"}
                else:
                    mongo_filter[mongo_field_name] = value
            else:
                logging.warning(f"Ignoring unknown query parameter: {param}")

        cursor = mongo_collection.find(mongo_filter)

        if limit_value is not None:
            try:
                limit_int = int(limit_value)
                if limit_int <= 0:
                    return jsonify({"error": "Limit must be a positive integer."}), 400
                cursor = cursor.limit(limit_int)
            except ValueError:
                return jsonify({"error": "Invalid limit value. Must be an integer."}), 400

        results = []
        for doc in cursor:
            if '_id' in doc and isinstance(doc['_id'], ObjectId):
                doc['documentId'] = str(doc['_id'])
                del doc['_id']
            if 'faceEmbedding' in doc:
                del doc['faceEmbedding']
            results.append(doc)

        return jsonify(results), 200

    except Exception as e:
        logging.error(f"An error occurred during document search: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- API Endpoint 2: Photo-based Search ---
@app.route('/search_by_photo', methods=['POST'])
def search_by_photo():
    if mongo_client is None:
        return jsonify({"error": "Database connection not established."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image file."}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported image format. Allowed: png, jpg, jpeg."}), 400

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, image_file.filename)
        image_file.save(temp_file_path)

        print(1)
        downscale(temp_file_path)

        logging.info(f"Saved uploaded image to: {temp_file_path}")
        print(2)
        query_embedding = get_face_embedding(temp_file_path)
        print(3)
        if query_embedding is None:
            return jsonify({"error": "Could not process image for face embedding. Ensure it contains exactly one clear face."}), 400

        limit = request.form.get('limit', 5, type=int)
        if limit <= 0:
            return jsonify({"error": "Limit must be a positive integer."}), 400

        search_results = search_mongodb_for_face(mongo_collection, query_embedding, limit)

        return jsonify(search_results), 200

    except Exception as e:
        logging.error(f"An error occurred during photo search: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    finally:
        # Clean up the temporary directory and its contents
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temporary directory: {temp_dir}")


# --- Running the Flask App ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=False)

