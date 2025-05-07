import os
import numpy as np
import shutil
from flask import Flask, request, render_template
from deepface import DeepFace

app = Flask(__name__)
UPLOAD_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDINGS_FILE = "facenet_embeddings1.npy"


@app.route("/", methods=["GET", "POST"])
def handle_images():
    if request.method == "POST":
        action = request.form.get("action")  
        
       
        if action == "add":
            try:
                person_name = request.form["person_name"]
                images = request.files.getlist("images")

                
                person_folder = os.path.join(app.config["UPLOAD_FOLDER"], person_name)
                os.makedirs(person_folder, exist_ok=True)

                for img in images:
                    img_path = os.path.join(person_folder, img.filename)
                    img.save(img_path)


                update_embeddings()

                return render_template("index.html", message=f"✅ {person_name} added & embeddings updated!")

            except Exception as e:
                return render_template("index.html", message=f"❌ Error: {str(e)}")

    
        elif action == "delete":
            try:
                person_name = request.form["person_name"]

                
                person_folder = os.path.join(app.config["UPLOAD_FOLDER"], person_name)
                if not os.path.exists(person_folder):
                    return render_template("index.html", message=f"❌ {person_name} not found!")

                
                shutil.rmtree(person_folder)

                
                if os.path.exists(EMBEDDINGS_FILE):
                    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
                    if person_name in database:
                        del database[person_name]
                        np.save(EMBEDDINGS_FILE, database)

                return render_template("index.html", message=f"✅ {person_name} deleted successfully!")

            except Exception as e:
                return render_template("index.html", message=f"❌ Error: {str(e)}")

    return render_template("index.html")  

# update embeddings code
def update_embeddings():
    database = {}

    if os.path.exists(EMBEDDINGS_FILE):
        database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

    for person in os.listdir(UPLOAD_FOLDER):
        person_path = os.path.join(UPLOAD_FOLDER, person)
        if os.path.isdir(person_path):
            embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    emb = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            if embeddings:
                database[person] = np.mean(embeddings, axis=0)

    np.save(EMBEDDINGS_FILE, database)
    print("✅ Face embeddings updated successfully!")

if __name__ == "__main__":
    app.run(debug=True)
