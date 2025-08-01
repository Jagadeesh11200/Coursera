import os
import json
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import re

# ------------------------
# Config
# ------------------------
EMBED_MODEL = "all-mpnet-base-v2"
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]
SCORE_MAPPING = {
    1: "Delivers much lower cognitive level than expected",
    2: "Mostly lower than expected, limited depth alignment",
    3: "Partial alignment",
    4: "Largely matches expected depth with minor mismatches",
    5: "Objectives fully match expected cognitive depth"
}
load_dotenv("api_key.env")

# ------------------------
# CDAS Evaluator
# ------------------------
class CognitiveDepthAlignmentEvaluator:
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)

    def cosine_similarity(self, dist1: List[float], dist2: List[float]) -> float:
        vec1 = np.array(dist1)
        vec2 = np.array(dist2)
        sim = util.cos_sim(vec1, vec2).item()
        return round(sim, 4)

    def map_score(self, similarity: float) -> Dict[str, float | str]:
        if similarity >= 0.90:
            score = 5
        elif similarity >= 0.75:
            score = 4
        elif similarity >= 0.55:
            score = 3
        elif similarity >= 0.35:
            score = 2
        else:
            score = 1
        return {"score": score, "label": SCORE_MAPPING[score]}

    def evaluate(self, expected_dist: List[float], actual_dist: List[float]) -> Dict:
        sim = self.cosine_similarity(expected_dist, actual_dist)
        result = self.map_score(sim)
        return {
            "similarity": sim,
            "score": result["score"],
            "label": result["label"]
        }

def generate_learner_feedback_per_lo(lo_text: str, expected_dist: List[float], actual_dist: List[float], similarity: float) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Step 1: Analysis prompt
    step1 = f"""
You are simulating the perspective of a learner evaluating whether a learning objective fits their goals.

Learning Objective:
"{lo_text}"

The course designers expected the following Bloom’s taxonomy depth distribution (based on course level):  
{expected_dist}

The actual depth distribution (as reflected in this objective) was:  
{actual_dist}

A model has calculated the alignment score between expected and actual as {similarity:.2f} (higher means better match).

Analyze how clearly the objective reflects the depth of learning promised at the course level. Comment on the clarity, challenge level, and whether it prepares learners for the expected cognitive level. Mention if it's too shallow, too ambitious, or well-aligned.
"""
    analysis = model.generate_content(step1).text.strip()

    # Step 2: Learner-facing comment prompt
    step2 = f"""
Write a short learner-facing comment (3–4 sentences) about whether this objective sets the right expectations and learning depth.

Tone and Style Guidelines:
- Use relaxed, learner-style phrasing — do not use "I" or "we"
- Prefer passive voice where possible
- Avoid expert tone or technical jargon
- Do not refer to alignment scores or distributions
- Focus on whether the objective is clear, useful, and mentally engaging

Analysis:
{analysis}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback



def generate_instructor_feedback_per_lo(lo_text: str, expected_dist: List[float], actual_dist: List[float], similarity: float) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Step 1: Structured analysis of alignment
    step1 = f"""
You are reviewing a learning objective for alignment with the intended Bloom’s taxonomy depth expectations.

--- Learning Objective ---
"{lo_text}"

--- Bloom’s Depth Comparison ---

- **Expected Distribution** (based on course level):  
  {expected_dist}

- **Observed Distribution** (based on LO wording):  
  {actual_dist}

- **Alignment Score** (cosine similarity):  
  {similarity:.2f} (higher means better alignment)

Step 1:  
Assess the quality of alignment using the information above. Focus on how well the LO supports the intended cognitive rigor (e.g., remembering, applying, creating), whether the wording accurately reflects this depth, and how clearly it guides learning outcomes.

Write a concise evaluation in two parts:
1. Highlight any strengths in clarity, depth, or alignment.
2. Point out gaps or mismatches, and suggest specific improvements to the LO phrasing or focus.
Limit response to 3–4 sentences.
"""
    analysis = model.generate_content(step1).text.strip()

    # Step 2: Instructor-facing feedback prompt
    step2 = f"""
Write a feedback message for the instructor based on the analysis below.

Guidelines:
- Avoid referencing scores or distributions directly.
- Keep the feedback practical and focused on observable issues in the LO.
- Tone: Professional, constructive, and to the point.
- Length: 3–4 sentences only.
- Do not include any opening lines or generic context.

Detailed Analysis:
{analysis}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback

# ------------------------
# Bloom Distribution Inference
# ------------------------
def get_expected_distribution(title: str, level: str, description: str) -> List[float]:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"You are an education expert. Based on the course title, level, and description,\n"
        f"predict the expected Bloom's Taxonomy level distribution as percentages for: {', '.join(BLOOM_LEVELS)}.\n"
        f"Return a valid JSON list of 5 floats adding to 1. Do not include 'Create'."
    )
    full_prompt = f"""
    Course Title: {title}
    Level: {level}
    Course Description: {description}

    {prompt}
    """
    response = model.generate_content(full_prompt)
    raw = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.IGNORECASE).strip()
    try:
        dist = json.loads(raw)
        if isinstance(dist, list) and len(dist) == 5 and abs(sum(dist) - 1.0) < 0.05:
            return dist
    except Exception:
        pass
    raise ValueError("Failed to generate expected Bloom's distribution")

def get_soft_distribution_per_lo(lo_text: str) -> List[float]:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        f"You are a Bloom's taxonomy expert. Classify the following learning objective into a soft Bloom's level distribution.\n"
        f"Return only a valid JSON list of 5 floats adding to 1, corresponding to: Remember, Understand, Apply, Analyze, Evaluate.\n"
        f"LO: \"{lo_text.strip()}\""
    )

    for attempt in range(2):  # Retry up to 2 times
        try:
            response = model.generate_content(prompt)
            raw = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.IGNORECASE).strip()

            # Attempt to extract a JSON list even if embedded in explanation
            match = re.search(r"\[(\s*\d*\.?\d+\s*,?)+\]", raw)
            if match:
                cleaned = match.group(0)
                dist = json.loads(cleaned)

                if isinstance(dist, list) and len(dist) == 5 and abs(sum(dist) - 1.0) < 0.05:
                    return dist
        except Exception as e:
            continue  # Retry

    # Fallback distribution (e.g., assume mostly Understand)
    print(f"[⚠️ Warning] Failed to parse Bloom distribution for:\n{lo_text}")
    return [0.05, 0.8, 0.1, 0.03, 0.02]

# ------------------------
# Metadata loader
# ------------------------
def extract_bloom_distributions(course_path: str) -> Dict:
    metadata_path = os.path.join(course_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.json not found in course folder")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    title = metadata.get("Course Title") or metadata.get("Course")
    level = metadata.get("Level", "Beginner")
    description = metadata.get("About this Course", "")

    evaluator = CognitiveDepthAlignmentEvaluator()
    expected = get_expected_distribution(title, level, description)

    results = {}

    for key, mod in metadata.items():
        if key.lower().startswith("module"):
            mod_name = mod.get("Name", key)
            objectives = mod.get("Learning Objectives", [])
            lo_results = {}
            for lo in objectives:
                try:
                    actual = get_soft_distribution_per_lo(lo)
                    result = evaluator.evaluate(expected, actual)
                    lo_results[lo] = {
                        "expected_distribution": [
                            {"level": BLOOM_LEVELS[i], "value": expected[i]} for i in range(5)
                        ],
                        "actual_distribution": [
                            {"level": BLOOM_LEVELS[i], "value": actual[i]} for i in range(5)
                        ],
                        **result
                    }
                except Exception as e:
                    lo_results[lo] = {"error": str(e)}

            results[mod_name] = lo_results

    return results

# ------------------------
# Main Entrypoint
# ------------------------
if __name__ == "__main__":
    BASE_PATH = r""
    OUTPUT_PATH = r""

    metadata_path = os.path.join(BASE_PATH, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    title = metadata.get("Course Title") or metadata.get("Course")
    level = metadata.get("Level", "Beginner")
    description = metadata.get("About this Course", "")
    expected = get_expected_distribution(title, level, description)

    evaluator = CognitiveDepthAlignmentEvaluator()
    all_results = {}  # Change from list to dict

    for key, mod in metadata.items():
        if key.lower().startswith("module"):
            module_name = mod.get("Name", key)
            learning_objectives = mod.get("Learning Objectives", [])
            module_results = []

            for lo_text in learning_objectives:
                actual = get_soft_distribution_per_lo(lo_text)
                result = evaluator.evaluate(expected, actual)

                learner_feedback = generate_learner_feedback_per_lo(lo_text, expected, actual, result["similarity"])
                instructor_feedback = generate_instructor_feedback_per_lo(lo_text, expected, actual, result["similarity"])

                module_results.append({
                    "learning_objective": lo_text,
                    "score": result["score"],
                    "label": result["label"],
                    "learner_feedback": learner_feedback,
                    "instructor_feedback": instructor_feedback
                })

            all_results[module_name] = module_results

    out_file = os.path.join(
        OUTPUT_PATH,
        os.path.basename(BASE_PATH) + "_cdas_per_objective_output new.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved result to: {out_file}")