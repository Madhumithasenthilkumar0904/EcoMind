
import os
import json
import sqlite3
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_retriever import retrieve_relevant_context

load_dotenv()

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
app.logger.setLevel(logging.INFO)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "waste_history.db")

client = genai.Client(api_key=API_KEY) if API_KEY else None

PROMPT_MODES = ["Zero-shot", "One-shot", "Few-shot"]

# Image upload settings
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# ---------------- DATABASE ----------------

def get_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as db:

        # Recommendations table
        db.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                waste_description TEXT NOT NULL,
                category TEXT,
                prompt_mode TEXT,
                recommendation TEXT
            )
        """)

        # Check existing recommendation columns
        columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(recommendations)"
            ).fetchall()
        }

        if "category" not in columns:
            db.execute(
                "ALTER TABLE recommendations ADD COLUMN category TEXT"
            )

        if "prompt_mode" not in columns:
            db.execute(
                "ALTER TABLE recommendations ADD COLUMN prompt_mode TEXT"
            )

        if "recommendation" not in columns:
            db.execute(
                "ALTER TABLE recommendations ADD COLUMN recommendation TEXT"
            )

        # Recommendation feedback table
        db.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recommendation_id INTEGER NOT NULL,
                useful INTEGER NOT NULL,
                category_correct INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recommendation_id)
                    REFERENCES recommendations(id)
                    ON DELETE CASCADE
            )
        """)

init_db()


# ---------------- RAG RETRIEVAL ----------------

def get_rag_context(waste):
    try:
        passages = retrieve_relevant_context(waste, n_results=4)

        context_parts = []
        sources = []

        for item in passages:
            source = item.get(
                "source",
                "Solid Waste Management Rules, 2026"
            )
            page = item.get("page", "Unknown")
            passage = item.get("text", "")

            if not passage:
                continue

            context_parts.append(
                f"Source: {source}\n"
                f"PDF page: {page}\n"
                f"Relevant passage:\n{passage}"
            )

            source_item = {
                "source": source,
                "page": page
            }

            if source_item not in sources:
                sources.append(source_item)

        return "\n\n".join(context_parts), sources

    except Exception:
        app.logger.exception("RAG retrieval failed")
        raise RuntimeError(
            "Could not retrieve information from the knowledge base. "
            "Check that the knowledge base has been built."
        )


# ---------------- GEMINI AND PROMPTS ----------------


def build_prompt(
    waste,
    category,
    mode,
    context="",
    rag_context=""
):
    examples = ""

    if mode == "One-shot":
        examples = """
Example:
Waste: Clean cardboard box
Category: Paper/Cardboard
Advice: Reuse as storage or recycle it if clean and dry.
Action: Reuse or place in paper recycling.
"""

    elif mode == "Few-shot":
        examples = """
Examples:
Waste: Banana peel
Category: Organic
Advice: Compost it where suitable.
Action: Compost.

Waste: Clean glass bottle
Category: Glass
Advice: Reuse it or take it to a suitable glass recycling facility.
Action: Reuse or recycle.

Waste: Used battery
Category: E-waste / Hazardous household waste
Advice: Keep it separate and use an approved battery collection point.
Action: Special collection; do not put it in ordinary recycling.
"""

    conversation_text = ""

    if context:
        conversation_text = f"""
CONVERSATION HISTORY:

{context}

Instructions:
- Understand the entire conversation.
- Treat the latest user message as a follow-up about the original waste item unless they clearly introduce a new item.
- Answer the latest question directly.
- Remember important details from earlier messages.
- Do not unnecessarily repeat the entire original recommendation.
- If the user asks another related question, answer it using the available context.
"""

    rag_text = ""

    if rag_context:
        rag_text = f"""
RETRIEVED KNOWLEDGE-BASE INFORMATION:

{rag_context}

Instructions for using retrieved information:
- Use relevant passages to guide your recommendation.
- Do not invent rules or requirements not supported by the retrieved passages.
- If the retrieved passages do not answer the question, clearly say that the available source does not specify it.
- Do not claim that a local municipal service is available unless the source confirms it.
"""

    return f"""
You are a practical waste-management and recycling advisor.

Analyze the waste description and provide safe, useful,
realistic guidance.

Do not invent local recycling services or claim that waste
has actually been recycled. Explain when local rules may differ.

Original waste description: {waste}
User's optional category: {category or "Not specified"}
Prompting method: {mode}

{examples}

{conversation_text}

{rag_text}

If essential information is missing, you may ask ONE short
follow-up question. Otherwise answer the user's latest question.

Return a JSON object with these keys:
{{
  "needs_follow_up": false,
  "follow_up_question": "",
  "category": "waste category",
  "advice": "clear answer or disposal guidance",
  "action": "main recommended next step"
}}

Return valid JSON only. Set needs_follow_up to true only
when more information is genuinely needed.
"""


def parse_model_json(raw):
    raw = (raw or "").strip()

    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)

        if isinstance(data, dict):
            return data

    except json.JSONDecodeError:
        pass

    return {
        "needs_follow_up": False,
        "follow_up_question": "",
        "category": "General waste",
        "advice": raw or "No recommendation was returned.",
        "action": "Check local waste-management guidance."
    }


def ask_gemini(prompt):
    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Check GEMINI_API_KEY "
            "in your .env file."
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text or ""


def save_recommendation(waste, category, mode, recommendation):
    now = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_db() as db:
        cursor = db.execute("""
            INSERT INTO recommendations
                (created_at, waste_description, category,
                 prompt_mode, recommendation)
            VALUES (?, ?, ?, ?, ?)
        """, (
            now,
            waste,
            category,
            mode,
            recommendation
        ))

        return cursor.lastrowid


# ---------------- IMAGE CLASSIFICATION ----------------
# ---------------- AI WASTE ADVISOR ----------------

@app.route("/get_advice", methods=["POST"])
def get_advice():
    data = request.get_json(silent=True) or {}

    waste = str(data.get("waste", "")).strip()
    category = str(data.get("category", "")).strip()
    mode = str(data.get("mode", "Zero-shot")).strip()

    # Accept both frontend context field names.
    context = str(
        data.get("conversation_context")
        or data.get("context")
        or ""
    ).strip()

    if not waste:
        return jsonify({
            "error": "Please enter a waste description."
        }), 400

    if mode not in PROMPT_MODES:
        mode = "Zero-shot"

    try:
        # Step 1: Retrieve relevant information using RAG.
        rag_context, sources = get_rag_context(waste)

        # Step 2: Build the Gemini prompt.
        prompt = build_prompt(
            waste=waste,
            category=category,
            mode=mode,
            context=context,
            rag_context=rag_context
        )

        # Step 3: Get the AI recommendation.
        raw = ask_gemini(prompt)
        result = parse_model_json(raw)

        # Step 4: Add source and mode information.
        result["sources"] = sources
        result["rag_used"] = bool(rag_context)
        result["mode"] = mode

        # Step 5: Save the recommendation in SQLite.
        record_id = save_recommendation(
            waste,
            result.get("category", category),
            mode,
            json.dumps(result, ensure_ascii=False)
        )

        result["id"] = record_id

        return jsonify(result)

    except Exception:
        app.logger.exception("AI waste recommendation failed")

        return jsonify({
            "error": (
                "Could not generate the recommendation. "
                "Check the VS Code terminal for details."
            )
        }), 502
def classify_waste_image(image_bytes, mime_type):
    """
    Identify waste from an uploaded image using Gemini Vision.
    """

    if client is None:
        raise RuntimeError(
            "Gemini API key not found. Check your .env file."
        )

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    prompt = """
You are an AI waste-image classification assistant.

Examine the uploaded image and identify the visible waste.

Return a JSON object with these keys:

{
  "description": "description of the visible waste",
  "category": "waste category",
  "confidence": "high, medium, or low"
}

Possible categories include:
- Organic waste
- Paper/Cardboard
- Plastic
- Glass
- Metal
- E-waste
- Textile
- Hazardous household waste
- Mixed waste
- General waste

Important instructions:
- Identify only what is reasonably visible.
- Do not assume the material if it cannot be determined.
- If the image is unclear or does not show waste,
  say so in the description and use low confidence.
- Do not claim that the item is recyclable merely because
  it belongs to a particular material category.
- Return valid JSON only.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image_part]
    )

    result = parse_model_json(response.text or "")

    return {
        "description": str(
            result.get("description", "Waste item identified from image.")
        ),
        "category": str(
            result.get("category", "General waste")
        ),
        "confidence": str(
            result.get("confidence", "low")
        )
    }


@app.route("/classify_image", methods=["POST"])
def classify_image():
    """
    Upload an image, identify the waste, retrieve RAG
    information, and generate a recommendation.
    """

    if "image" not in request.files:
        return jsonify({
            "error": "Please select an image to upload."
        }), 400

    image = request.files["image"]

    if not image.filename:
        return jsonify({
            "error": "Please select an image."
        }), 400

    if image.mimetype not in ALLOWED_IMAGE_TYPES:
        return jsonify({
            "error": "Only JPG, PNG and WEBP images are supported."
        }), 400

    try:
        image_bytes = image.read(MAX_IMAGE_SIZE + 1)

        if not image_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        if len(image_bytes) > MAX_IMAGE_SIZE:
            return jsonify({
                "error": "Image size must not exceed 5 MB."
            }), 413

        # Step 1: Identify waste using Gemini Vision.
        identification = classify_waste_image(
            image_bytes,
            image.mimetype
        )

        waste = identification["description"]
        category = identification["category"]

        # Do not generate disposal advice for an unclear image.
        if identification["confidence"].lower() == "low":
            return jsonify({
                "error": (
                    "The image is unclear. Please upload a clearer "
                    "image showing the waste item."
                ),
                "identification": identification
            }), 422

        # Step 2: Retrieve relevant rules using RAG.
        rag_context, sources = get_rag_context(
            f"{waste}. Waste category: {category}"
        )

        # Step 3: Generate disposal recommendation.
        prompt = build_prompt(
            waste=waste,
            category=category,
            mode="Zero-shot",
            rag_context=rag_context
        )

        raw = ask_gemini(prompt)
        result = parse_model_json(raw)

        if result.get("needs_follow_up"):
            return jsonify({
                "error": (
                    "The image was identified, but more information "
                    "is needed. Please describe the item in the "
                    "text-based advisor."
                ),
                "identification": identification
            }), 422

        result["needs_follow_up"] = False
        result["sources"] = sources
        result["rag_used"] = bool(rag_context)
        result["mode"] = "Image Classification"

        # Step 4: Save the recommendation in existing history.
        record_id = save_recommendation(
            waste,
            result.get("category", category),
            "Image Classification",
            json.dumps(result, ensure_ascii=False)
        )

        result["id"] = record_id
        result["identification"] = identification

        return jsonify(result)

    except Exception:
        app.logger.exception("Image classification failed")

        return jsonify({
            "error": (
                "Image classification failed. "
                "Check the VS Code terminal for details."
            )
        }), 502


# ---------------- PROMPT EVALUATION ----------------

def heuristic_score(result):
    """
    A completeness heuristic, not a measure of factual accuracy.
    """

    text = (
        str(result.get("advice", "")) + " " +
        str(result.get("action", ""))
    ).lower()

    checks = [
        bool(result.get("category")),
        bool(result.get("action")),
        any(
            word in text
            for word in ["reuse", "recycle", "compost", "dispose"]
        ),
        any(
            word in text
            for word in ["because", "avoid", "keep", "separate"]
        ),
        len(text.split()) >= 20
    ]

    return sum(20 for passed in checks if passed)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(silent=True) or {}

    waste = str(data.get("waste", "")).strip()
    category = str(data.get("category", "")).strip()

    if not waste:
        return jsonify({
            "error": "Enter a waste description to evaluate."
        }), 400

    results = []

    try:
        for mode in PROMPT_MODES:
            raw = ask_gemini(
                build_prompt(waste, category, mode)
            )

            result = parse_model_json(raw)

            results.append({
                "mode": mode,
                "score": heuristic_score(result),
                "result": result
            })

        return jsonify({
            "results": results,
            "note": (
                "Scores are heuristic completeness indicators, "
                "not verified measures of factual accuracy "
                "or environmental impact."
            )
        })

    except Exception:
        app.logger.exception("Prompt evaluation failed")

        return jsonify({
            "error": (
                "Prompt evaluation failed. "
                "Check the VS Code terminal."
            )
        }), 502


# ---------------- RAG VS NON-RAG EVALUATION ----------------

@app.route("/evaluate_rag", methods=["POST"])
def evaluate_rag():
    data = request.get_json(silent=True) or {}

    waste = str(data.get("waste", "")).strip()
    category = str(data.get("category", "")).strip()
    mode = str(data.get("mode", "Zero-shot")).strip()

    if not waste:
        return jsonify({
            "error": "Enter a waste description to evaluate."
        }), 400

    if mode not in PROMPT_MODES:
        mode = "Zero-shot"

    try:
        normal_raw = ask_gemini(
            build_prompt(waste, category, mode)
        )
        normal_result = parse_model_json(normal_raw)

        rag_context, sources = get_rag_context(waste)

        rag_raw = ask_gemini(
            build_prompt(
                waste,
                category,
                mode,
                rag_context=rag_context
            )
        )
        rag_result = parse_model_json(rag_raw)

        return jsonify({
            "mode": mode,
            "without_rag": {
                "score": heuristic_score(normal_result),
                "result": normal_result
            },
            "with_rag": {
                "score": heuristic_score(rag_result),
                "result": rag_result,
                "sources": sources
            },
            "note": (
                "The scores indicate response completeness only. "
                "They do not establish factual accuracy. "
                "Review both answers against the cited rules "
                "to assess correctness."
            )
        })

    except Exception:
        app.logger.exception("RAG comparison failed")

        return jsonify({
            "error": (
                "RAG comparison failed. "
                "Check the VS Code terminal."
            )
        }), 502

# ---------------- RECOMMENDATION FEEDBACK ----------------

@app.route("/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json(silent=True) or {}

    recommendation_id = data.get("recommendation_id")
    useful = data.get("useful")
    category_correct = data.get("category_correct")

    if not isinstance(recommendation_id, int) or isinstance(recommendation_id, bool):
        return jsonify({"error": "Invalid recommendation ID."}), 400

    if useful not in (True, False, 1, 0):
        return jsonify({"error": "Please provide valid feedback."}), 400

    if category_correct not in (None, True, False, 1, 0):
        return jsonify({"error": "Invalid category feedback."}), 400

    try:
        with get_db() as db:
            exists = db.execute(
                "SELECT id FROM recommendations WHERE id = ?",
                (recommendation_id,)
            ).fetchone()

            if not exists:
                return jsonify({"error": "Recommendation not found."}), 404

            db.execute("""
                INSERT INTO recommendation_feedback
                    (recommendation_id, useful, category_correct, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                recommendation_id,
                int(useful),
                None if category_correct is None else int(category_correct),
                datetime.now().astimezone().isoformat(timespec="seconds")
            ))

        return jsonify({"message": "Feedback saved successfully."})

    except Exception:
        app.logger.exception("Feedback submission failed")
        return jsonify({"error": "Could not save feedback."}), 500
# ---------------- HISTORY ----------------

@app.route("/history", methods=["GET"])
def history():
    try:
        with get_db() as db:
            rows = db.execute("""
                SELECT id, created_at, waste_description, category,
                       prompt_mode, recommendation
                FROM recommendations
                ORDER BY id DESC
                LIMIT 100
            """).fetchall()

        items = []

        for row in rows:
            item = dict(row)

            try:
                item["recommendation"] = json.loads(
                    item["recommendation"] or "{}"
                )
            except (json.JSONDecodeError, TypeError):
                pass

            items.append(item)

        return jsonify({"history": items})

    except Exception:
        app.logger.exception("History request failed")

        return jsonify({
            "error": "Could not load history."
        }), 500


@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_history_item(record_id):
    with get_db() as db:
        cursor = db.execute(
            "DELETE FROM recommendations WHERE id = ?",
            (record_id,)
        )

    if cursor.rowcount == 0:
        return jsonify({
            "error": "History record not found."
        }), 404

    return jsonify({
        "message": "History record deleted."
    })


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with get_db() as db:
        db.execute("DELETE FROM recommendations")

    return jsonify({
        "message": "History cleared."
    })


# ---------------- SUSTAINABILITY INSIGHTS ----------------

@app.route("/dashboard", methods=["GET"])
def dashboard_data():
    try:
        with get_db() as db:
            total = db.execute("""
                SELECT COUNT(*) AS total
                FROM recommendations
            """).fetchone()["total"]

            unique_waste = db.execute("""
                SELECT COUNT(
                    DISTINCT LOWER(TRIM(waste_description))
                ) AS total
                FROM recommendations
            """).fetchone()["total"]

            
            raw_category_rows = db.execute("""
                SELECT category, COUNT(*) AS total
                FROM recommendations
                GROUP BY category
            """).fetchall()

            category_totals = {
                "Organic Waste": 0,
                "Plastic Waste": 0,
                "Paper & Cardboard": 0,
                "Glass Waste": 0,
                "Metal Waste": 0,
                "E-waste": 0,
                "Hazardous Waste": 0,
                "Textile Waste": 0,
                "Mixed Waste": 0,
                "Other / Unspecified": 0
            }

            for row in raw_category_rows:
                name = (row["category"] or "").strip().lower()
                count = row["total"]

                if not name or "not specified" in name or "uncategorized" in name:
                    standard = "Other / Unspecified"
                elif any(word in name for word in ["e-waste", "electronic", "electronics"]):
                    standard = "E-waste"
                elif "hazard" in name or "toxic" in name:
                    standard = "Hazardous Waste"
                elif any(word in name for word in ["organic", "food", "biodegradable", "garden"]):
                    standard = "Organic Waste"
                elif any(word in name for word in ["plastic", "polythene"]):
                    standard = "Plastic Waste"
                elif any(word in name for word in ["paper", "cardboard"]):
                    standard = "Paper & Cardboard"
                elif "glass" in name:
                    standard = "Glass Waste"
                elif any(word in name for word in ["metal", "aluminium", "aluminum", "steel"]):
                    standard = "Metal Waste"
                elif any(word in name for word in ["textile", "fabric", "cloth"]):
                    standard = "Textile Waste"
                elif "mixed" in name:
                    standard = "Mixed Waste"
                else:
                    standard = "Other / Unspecified"

                category_totals[standard] += count

            category_rows = [
                {"label": name, "total": count}
                for name, count in category_totals.items()
                if count > 0
            ]

            category_rows.sort(key=lambda item: (-item["total"], item["label"]))


            
            raw_mode_rows = db.execute("""
                SELECT prompt_mode, COUNT(*) AS total
                FROM recommendations
                GROUP BY prompt_mode
            """).fetchall()

            mode_totals = {
                "Zero-shot": 0,
                "One-shot": 0,
                "Few-shot": 0,
                "Image Classification": 0
            }

            for row in raw_mode_rows:
                name = (row["prompt_mode"] or "").strip().lower()
                count = row["total"]

                if "image" in name:
                    standard = "Image Classification"
                elif "few" in name:
                    standard = "Few-shot"
                elif "one" in name:
                    standard = "One-shot"
                elif "zero" in name:
                    standard = "Zero-shot"
                else:
                    standard = "Zero-shot"

                mode_totals[standard] += count

            mode_rows = [
                {"label": name, "total": count}
                for name, count in mode_totals.items()
                if count > 0
            ]

            mode_rows.sort(key=lambda item: (-item["total"], item["label"]))


            daily_rows = db.execute("""
                SELECT
                    SUBSTR(created_at, 1, 10) AS day,
                    COUNT(*) AS total
                FROM recommendations
                WHERE created_at IS NOT NULL
                GROUP BY SUBSTR(created_at, 1, 10)
                ORDER BY day DESC
                LIMIT 7
            """).fetchall()

                        # Recent recommendation records
            recent_rows = db.execute("""
                SELECT id, created_at, waste_description,
                       category, prompt_mode
                FROM recommendations
                ORDER BY id DESC
                LIMIT 5
            """).fetchall()

            # Feedback statistics
            feedback_row = db.execute("""
                SELECT
                    COUNT(*) AS total_feedback,
                    SUM(CASE WHEN useful = 1 THEN 1 ELSE 0 END)
                        AS positive_feedback,
                    SUM(CASE WHEN useful = 0 THEN 1 ELSE 0 END)
                        AS negative_feedback,
                    SUM(CASE WHEN category_correct = 1 THEN 1 ELSE 0 END)
                        AS correct_categories,
                    SUM(CASE WHEN category_correct = 0 THEN 1 ELSE 0 END)
                        AS incorrect_categories
                FROM recommendation_feedback
            """).fetchone()

        categories = [
            {
                "label": row["label"],
                "total": row["total"]
            }
            for row in category_rows
        ]

        modes = [
            {
                "label": row["label"],
                "total": row["total"]
            }
            for row in mode_rows
        ]

        daily = [
            {
                "day": row["day"],
                "total": row["total"]
            }
            for row in reversed(daily_rows)
        ]

        most_common = (
            categories[0]["label"]
            if categories else "No data yet"
        )

        return jsonify({
            "total_recommendations": total,
            "unique_waste_descriptions": unique_waste,
            "most_common_category": most_common,
            "categories": categories,
            "prompt_modes": modes,
            "daily_activity": daily,
            "recent_activity": [
                dict(row) for row in recent_rows
            ],
                        "feedback_stats": {
                "total_feedback": feedback_row["total_feedback"] or 0,
                "positive_feedback": feedback_row["positive_feedback"] or 0,
                "negative_feedback": feedback_row["negative_feedback"] or 0,
                "correct_categories": feedback_row["correct_categories"] or 0,
                "incorrect_categories": feedback_row["incorrect_categories"] or 0
            },
            "note": (
                "These figures summarize recommendations "
                "saved by the app. They do not measure actual "
                "recycling, waste diverted, or environmental impact."
            )
        })

    except Exception:
        app.logger.exception("Dashboard data request failed")

        return jsonify({
            "error": "Could not load dashboard data."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
