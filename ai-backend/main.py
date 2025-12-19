from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import tempfile, shutil, uuid, os

# 🔹 Gemini imports
import google.generativeai as genai

# ------------------ APP SETUP ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GEMINI SETUP ------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def explain_match(confidence: float) -> str:
    """
    Uses Gemini to explain the AI confidence score
    in human-friendly and ethical terms.
    """
    prompt = f"""
    An AI face recognition system produced a similarity score of {confidence}%.
    Explain this in simple, human-friendly language.
    Add an ethical disclaimer.
    Encourage human verification.
    Keep it short (3–4 sentences).
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return (
            "This score indicates a possible similarity. "
            "Results may vary due to lighting, angle, or image quality. "
            "AI outputs are indicative only—human verification is recommended."
        )

# ------------------ TEMP IN-MEMORY DB (MVP) ------------------

missing_persons = []

def save_temp_image(upload_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    shutil.copyfileobj(upload_file.file, temp)
    return temp.name

# ------------------ REPORT MISSING ------------------

@app.post("/report-missing")
async def report_missing(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    image: UploadFile = File(...)
):
    img_path = save_temp_image(image)

    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        enforce_detection=True
    )[0]["embedding"]

    person_id = str(uuid.uuid4())

    missing_persons.append({
        "id": person_id,
        "name": name,
        "age": age,
        "gender": gender,
        "embedding": embedding,
        "image": img_path
    })

    return {"status": "reported", "id": person_id}

# ------------------ MATCH PERSON ------------------

@app.post("/match")
async def match_person(image: UploadFile = File(...)):
    img_path = save_temp_image(image)

    query_embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        enforce_detection=True
    )[0]["embedding"]

    results = []

    for person in missing_persons:
        result = DeepFace.verify(
            img1_path=img_path,
            img2_path=person["image"],
            enforce_detection=False
        )

        confidence = round(result["confidence"], 2)

        results.append({
            "id": person["id"],
            "name": person["name"],
            "confidence": confidence,
            "explanation": explain_match(confidence)
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {"matches": results}
