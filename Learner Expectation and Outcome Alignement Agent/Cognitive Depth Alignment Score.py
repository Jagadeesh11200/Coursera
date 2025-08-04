import os
import json
import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import re

# ------------------------
# Config
# ------------------------
EMBED_MODEL = "all-mpnet-base-v2"
BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate"]
# SCORE_MAPPING = {
#     1: "Delivers much lower cognitive level than expected",
#     2: "Mostly lower than expected, limited depth alignment",
#     3: "Partial alignment",
#     4: "Largely matches expected depth with minor mismatches",
#     5: "Objectives fully match expected cognitive depth"
# }
load_dotenv("api_key.env")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-001")

# ------------------------
# CDAS Evaluator
# ------------------------
class CognitiveDepthAlignmentEvaluator:
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)

    def extract_json(self, text):
        try:
            json_match = re.search(r'{.*}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None

    def cosine_similarity(self, dist1: List[float], dist2: List[float]) -> float:
        vec1 = np.array(dist1)
        vec2 = np.array(dist2)
        sim = util.cos_sim(vec1, vec2).item()
        return round(sim, 4)

    def generate_score_with_label_from_gemini(self, similarity: float) -> Dict[str, Any]:
        prompt = f"""
    You are evaluating how closely two cognitive depth profiles align using a cosine similarity score.

    - Cosine similarity: {similarity:.3f}

    Rate this alignment using a Cognitive Depth Alignment Score (CDAS), where:
    1.0 → Very poor alignment (objectives are far below expected cognitive depth)
    5.0 → Excellent alignment (objectives fully match expected depth)
    Allow decimal scores (e.g., 3.7, 4.2, etc.).

    Also provide a one-line justification (label) for the score that explains the alignment.

    Output format (JSON):
    {{
    "score": <decimal between 1.0 and 5.0>,
    "label": "<one-line justification>"
    }}
    """

        response = model.generate_content(prompt)
        parsed = self.extract_json(response.text)
        if parsed:
            return parsed
        else:
            return {
                "score": round(5 * similarity, 2),
                "label": "Proportional alignment based on cosine similarity"
            }

    def evaluate(self, expected_dist: List[float], actual_dist: List[float]) -> Dict:
        sim = self.cosine_similarity(expected_dist, actual_dist)
        result = self.generate_score_with_label_from_gemini(sim)
        return {
            "similarity": sim,
            "score": result["score"],
            "label": result["label"]
        }

def generate_learner_feedback_per_lo(lo_text: str, expected_dist: List[float], actual_dist: List[float], similarity: float) -> str:

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

def generate_module_feedback_cdas(module_name, avg_score, persona_type: str):
    if persona_type == 'learner':
        learner_prompt = f"""
Persona: You are simulating a learner reflecting on a module's cognitive depth alignment. Reflect as someone who has just completed the module and is evaluating how well the learning objectives matched the expected depth of thinking.

As a learner, consider how challenging and mentally engaging this module was. Think about whether the activities and objectives made you recall facts, apply ideas, analyze information, or go even deeper—and whether that matched what you thought you'd be doing based on the course’s overall expectations.

**Internal Evaluation Score (not to be mentioned):**
- Alignment Score: {avg_score} (Scale 1–5)

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, linguistic accessibility, and how comprehensively topics were covered.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt:**
Write a reflection from a learner’s perspective on whether the module "{module_name}" delivered the right level of mental challenge and cognitive engagement, given an internal alignment score of {avg_score}.
"""
        response = model.generate_content(learner_prompt)
        return response.text.strip()

    elif persona_type == 'instructor':
        instructor_prompt = f"""
You are a curriculum evaluator reviewing the module "{module_name}" in a course. Provide a short evaluation of how well the learning objectives in this module align with the level of cognitive engagement expected from the course's scope and level.

Your insights are based on a Cognitive Depth Alignment Score (CDAS), which evaluates whether the objectives promote thinking at the appropriate cognitive depth.

**Internal Reference:**
- Alignment Score: {avg_score} (Scale 1–5)

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, based solely on the underlying quality signals.
* **Avoid:** Direct references to metric names, numerical values, or overly technical instructional design language.

**Prompt:**
Write feedback for the instructor of the module "{module_name}" based on a CDAS alignment score of {avg_score}, commenting on how well the objectives support cognitive depth and offering a constructive suggestion if relevant.
"""
        response = model.generate_content(instructor_prompt)
        return response.text.strip()

    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")

def generate_course_feedback_cdas(course_name, avg_score, persona_type: str):
    if persona_type == 'learner':
        learner_prompt = f"""
Persona: You are simulating a learner's direct experience of a course's cognitive depth alignment. Reflect as someone who has just completed the course and is evaluating how well the course’s learning objectives matched the depth and type of cognitive engagement you expected.

As a learner, consider how appropriately challenging and mentally engaging the course was, based on the way learning objectives were structured. Think about whether the content met, exceeded, or fell short of what was implied in the course overview and how aligned the modules felt with what was promised.

**Evaluation Methodology (behind the scenes):**
* **Alignment Score (Range: 0–5):** AI-derived judgment of how well the cognitive expectations set in the course description are reflected and scaffolded in the actual learning objectives.

**Do not mention these terms or scores in the reflection. Just simulate a real learner's perceived experience.**


- Alignment Score: {avg_score}

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, preparedness, and how the prerequisites impacted the ease of transitioning into course content.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt:**
Given an alignment rating of {avg_score}, write a short learner-style reflection on whether the cognitive expectations set in the course matched the actual learning experience in terms of mental challenge and depth.
"""
        response = model.generate_content(learner_prompt)
        return response.text.strip()

    elif persona_type == 'instructor':
        instructor_prompt = f"""
You are a pedagogical reviewer providing feedback to the course instructor based on a Cognitive Depth Alignment Score analysis. Your goal is to help them evaluate whether the learning objectives throughout the course appropriately align with the implied depth of engagement in the course description.

**Evaluation Methodology (behind the scenes):**
* **Alignment Score (Range: 0–5):** Measures how well the course scaffolds learning objectives to match expected cognitive processes (e.g., recall, analysis, synthesis).

**These are the internal scores guiding your response:**
- Alignment Score: {avg_score}

Your response should adhere to the following guidelines:
    * **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
    * **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
    * **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
    * **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to how prerequisites are framed, integrated, or reinforced in the course.
    * **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt:**
Based on the above cognitive similarity and alignment score, write a short, data-informed assessment of how well the course learning objectives align with the expected cognitive depth of the course "{course_name}". Give one clear, actionable suggestion to improve alignment if needed.
"""
        response = model.generate_content(instructor_prompt)
        return response.text.strip()

    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")

# ------------------------
# Bloom Distribution Inference
# ------------------------
def get_expected_distribution(title: str, level: str, description: str) -> List[float]:
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

    # Overall structure
    all_results = {
        "Course": title,
        "Course Level CDAS Score": None,
        "Course Level Feedback": None,
        "Modules": {}
    }

    all_module_scores = []

    for key, mod in metadata.items():
        if key.lower().startswith("module"):
            module_name = mod.get("Name", key)
            learning_objectives = mod.get("Learning Objectives", [])
            module_results = []

            module_scores = []

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

                module_scores.append(result["score"])

            module_avg_score = round(sum(module_scores) / len(module_scores), 2) if module_scores else 0.0
            all_module_scores.append(module_avg_score)

            module_instructor_feedback = generate_module_feedback_cdas(module_name, module_avg_score, persona_type="instructor") 
            module_learner_feedback = generate_module_feedback_cdas(module_name, module_avg_score, persona_type="learner")

            all_results["Modules"][module_name] = {
                "Module Level CDAS Score": module_avg_score,
                "Instructor Feedback": module_instructor_feedback,
                "Learner Feedback": module_learner_feedback,
                "Objectives": module_results
            }

    # Course-level score and feedback
    course_avg_score = round(sum(all_module_scores) / len(all_module_scores), 2) if all_module_scores else 0.0
    course_learner_feedback = generate_course_feedback_cdas(title, avg_score=course_avg_score, persona_type="learner")
    course_instructor_feedback = generate_course_feedback_cdas(title, avg_score=course_avg_score, persona_type="instructor")

    all_results["Course Level CDAS Score"] = course_avg_score
    all_results["Course Level Feedback"] = {
        "Learner Feedback": course_learner_feedback,
        "Instructor Feedback": course_instructor_feedback
    }

    out_file = os.path.join(
        OUTPUT_PATH,
        os.path.basename(BASE_PATH) + "_cdas_per_objective_output.json"
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved result to: {out_file}")