import os
import json
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import transformers
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import argparse
import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
import ast

# Load NLI model once (at global scope)
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load API Key
load_dotenv('api_key.env')
genai.configure(api_key="") # Ensure API key is loaded correctly from .env
model = genai.GenerativeModel('gemini-2.0-flash-001')

# BASE_PATH setup
BASE_PATH = r""
METADATA_PATH = r""
OUTPUT_PATH = r""
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load NLP tools
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_concepts_prompt(course_name: str, about_course: str, module_name: str, original_lo: str) -> str:
    return f"""
You are a curriculum designer. Based on the following course metadata, extract **at most 5** high-level concepts or themes that best represent what this course teaches.

**Course Name:** {course_name}

**About this Course:**
\"\"\"
{about_course}
\"\"\"

**Module Title:**
{module_name}

**A Learning Objective to Ground On:**
{original_lo}

Guidelines:
- Extract *at most 5* core concepts that reflect the central skills or knowledge areas covered in the module.
- Each concept should be short and specific (1–4 words), e.g., "Model Evaluation", "Prompt Engineering".
- Avoid redundancy or vague topics like "Learning" or "Understanding".
- Do not include examples or detailed explanations.
- Return the concepts as a plain bullet list.
"""


def extract_concepts(course_name, about_text, module_name, single_lo):
    prompt = generate_concepts_prompt(course_name, about_text, module_name, single_lo)
    response = model.generate_content(prompt)
    return [line.strip('-* \n') for line in response.text.splitlines() if line.strip()]

def assign_concepts_and_bloom(syllabus_points, concepts):
    prompt = f"""
You are a curriculum expert.

Given:
- A list of extracted **course concepts**
- A list of **syllabus points** for a course module

For each syllabus point, do the following:
1. Assign the **most relevant concept** from the list.
2. Assign an appropriate **Bloom's Taxonomy level** from: Remember, Understand, Apply, Analyze.

Your output must be a JSON list. Each item must include:
- "Syllabus Point"
- "Concept"
- "Bloom Level"

Here is the input:

"Concepts":
{json.dumps(concepts, indent=2)}

"Syllabus Points":
{json.dumps(syllabus_points, indent=2)}

Output format:
[
  {{
    "Syllabus Point": "Video: Transformer architecture",
    "Concept": "Transformer Architecture",
    "Bloom Level": "Understand"
  }},
  ...
]
Return only the JSON list. Do not include commentary or explanations.
"""
    response = model.generate_content(prompt)
    try:
        clean_text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        return json.loads(clean_text)
    except Exception:
        print("⚠️ Error parsing JSON from Gemini response (Concept + Bloom).")
        print("Response was:", response.text)
        return []

def generate_learning_objectives(assignments):
    prompt = f"""
You are an expert instructional designer.

Your task is to generate concise, measurable, and high-quality learning objectives using the provided syllabus point, concept, and Bloom's Taxonomy level.

Instructions:
- Generate one learning objective per syllabus point.
- Each objective must:
  - Use the Bloom level to guide a suitable action verb.
  - Meaningfully incorporate the related concept (do not simply repeat the syllabus point).
  - Be phrased clearly and academically, in one sentence.
  - Start learning objectives with actionable verbs based on Bloom's Taxonomy, such as Identify, Define, Describe, Analyze, Evaluate, and Create, rather than using phrases like 'Students will be able to.'

Input:
{json.dumps(assignments, indent=2)}

Output format:
[
  {{
    "Syllabus Point": "...",
    "Concept": "...",
    "Bloom Level": "...",
    "Learning Objective": "..."
  }},
  ...
]

Return only the JSON list.
"""
    response = model.generate_content(prompt)
    try:
        clean_text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        return json.loads(clean_text)
    except Exception:
        print("⚠️ Error parsing JSON from Gemini response (Learning Objectives).")
        print("Response was:", response.text)
        return []


def gemini_similarity_and_structure(org_lo, gen_lo):
    prompt = f"""
    You are an education quality analyst. Evaluate the following Original Learning Objective (LO) compared to the Generated Ideal LO.

    Original LO: "{org_lo}"
    Generated LO: "{gen_lo}"

    1. Determine the similarity and coverage: Does the original LO fully or partially capture the intent and scope of the generated LO?
    2. Check Bloom's Taxonomy verbs: Are both LOs using comparable cognitive levels (e.g., Remember, Understand, Apply, Analyze, Evaluate, Create)?
    3. Output a float rating from 1.00 to 5.00 based on their semantic and structural alignment.

    Respond in this exact format:
    SimilarityCoverageScore: <float between 1.00 to 5.00>
    BloomMatch: <Yes/Partial/No>
    Reason: <brief explanation>
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_similarity_and_structure(response_text):
    sim_score = re.search(r"SimilarityCoverageScore:\s*([0-9]+\.[0-9]+)", response_text)
    bloom_match = re.search(r"BloomMatch:\s*(Yes|Partial|No)", response_text)
    reason = re.search(r"Reason:\s*(.*)", response_text, re.DOTALL)
    return {
        "similarity_coverage_score": float(sim_score.group(1)) if sim_score else 0.0,
        "bloom_taxonomy_match": bloom_match.group(1) if bloom_match else "Unknown",
        "reasoning": reason.group(1).strip() if reason else ""
    }

def gemini_final_rating(org_lo, gen_lo):
    prompt = f"""
    As a learning objective evaluator, provide a rating from 1.00 to 5.00 of the Original LO based on how well it aligns in meaning, depth, and educational value with the Generated LO.

    Original LO: "{org_lo}"
    Generated LO: "{gen_lo}"

    Respond in format:
    Rating: <float between 1.00 to 5.00>
    Reason: <brief explanation>
    """
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_final_rating(response_text):
    match = re.search(r"Rating:\s*([0-9]+\.[0-9]+)", response_text)
    reason = re.search(r"Reason:\s*(.*)", response_text, re.DOTALL)
    return float(match.group(1)) if match else 0.0, reason.group(1).strip() if reason else ""

def generate_individual_lo_feedback(module_name, original_lo, gen_lo, cosine_score, coverage_score, nli_score, bloom_match, final_score, gemini_reasoning, gemini_structure_reason, hybrid_weighted_score, persona_type: str):
    if persona_type == 'learner':
        step1 = f"""
You are simulating the perspective of a learner deciding whether to take a course module titled "{module_name}".

Original Learning Objective:
"{original_lo}"

Most Relevant Generated Objective (AI-generated for validation):
"{gen_lo}"

AI is helping assess whether the module stays true to what it claims to teach. It checks how clearly the goals are expressed, how well key ideas are covered, and whether the depth of learning matches expectations.

Evaluation Metrics (scale: 0–5):

- **Semantic Match**: {cosine_score}
  → Measures how close the generated goal's meaning is to the original. Higher means the same ideas are preserved.

- **Content Coverage**: {coverage_score}
  → Checks if important concepts from the original are included. Higher means better topical match.

- **Logical Alignment**: {nli_score}
  → Measures how strongly the generated objective is inferred or entailed by the original. Higher values indicate better logical agreement.

- **Depth Match**: {bloom_match}
  → Evaluates whether the learning challenge level (recall vs critical thinking) is aligned.

- **AI Judgment**: {final_score}
  → An LLM-based assessment of quality, tone, and accuracy of the rewritten goal.

- **Final Match Score**: {hybrid_weighted_score}
  → A combined score summarizing the above signals into an overall alignment estimate.

Gemini Content Analysis:
{gemini_reasoning}

Gemini Structure Analysis:
{gemini_structure_reason}

From the learner's point of view, decide whether the alignment is strong, moderate, or weak. Mention which aspects (meaning, coverage, depth, clarity) support or weaken confidence in what will actually be learned.
"""
        analysis = model.generate_content(step1).text.strip()

        step2 = f"""
Write a short learner-facing comment (3–4 sentences) about whether the module objective is clear and worth learning from.

To guide tone and style, think of how a learner migh describe the experience.

Requirements:
- Use learner-style phrasing, but without using "I" or "we"
- Passive voice should be used where possible
- Avoid expert tone or generic filler like "it feels like" or "seems to be"
- Language should sound natural, relaxed, and confident — not overly formal
- Focus on clarity, usefulness, and whether learning can actually happen based on the objective

Analysis:
{analysis}
"""
        feedback = model.generate_content(step2).text.strip()
        return feedback
    elif persona_type == 'instructor':
        step1 = f"""
You are reviewing a learning objective match for the course module titled "{module_name}".

--- Learning Objectives ---

Original LO:
"{original_lo}"

Best Matching Generated LO:
"{gen_lo}"

--- Evaluation Metrics (0 to 5) ---

- **Semantic Closeness**: {cosine_score}
  Assesses how well the meaning of the original LO is preserved.

- **Content Representation**: {coverage_score}
  Checks if major ideas and skills in the original LO are reflected.

- **Logical Alignment**: {nli_score}
  → Measures how strongly the generated objective is inferred or entailed by the original. Higher values indicate better logical agreement.

- **Cognitive Depth Alignment**: {bloom_match}
  Determines if the level of learning expected (e.g., recall vs analysis) is maintained.

- **Language Model Evaluation**: {final_score}
  Gauges the overall relevance, clarity, and instructional soundness of the generated LO.

- **Final Weighted Score**: {hybrid_weighted_score}
  Summary score reflecting overall alignment strength.

--- Gemini Model Explanations ---

- **Semantic Reasoning**:
{gemini_reasoning}

- **Structural Feedback**:
{gemini_structure_reason}

Step 1:
Assess the alignment quality using the metrics and reasoning above. Focus on how well the rewritten objective captures the intended meaning, covers core concepts, maintains the expected cognitive challenge, and follows good instructional structure.

Write a concise evaluation with two parts:
1. Strengths observed in meaning, coverage, depth, and structure.
2. Specific gaps or weaknesses, followed by concrete improvement suggestions.
Total: 3–4 sentences only.
"""
        analysis = model.generate_content(step1).text.strip()

        step2 = f"""
Write a feedback message for the instructor based on the analysis below.

Guidelines:
- Do NOT mention metrics directly.
- Focus only on observable strengths and areas of improvement.
- Tone: Constructive, focused, and professional.
- Length: Strictly 3–4 sentences.
- Strictly do not include any introductory statements.

Detailed Analysis:
{analysis}
"""
        feedback = model.generate_content(step2).text.strip()
        return feedback
    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")


def nli_entailment_score(premise, hypothesis):
    try:
        result = nli_model(
            sequences=premise,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="This example is {}: " + hypothesis
        )
        for label, score in zip(result["labels"], result["scores"]):
            if label.lower() == "entailment":
                return score
        return 0.0
    except Exception as e:
        print("NLI error:", e)
        return 0.0

def bidirectional_nli_score(lo1, lo2):
    score1 = nli_entailment_score(lo1, lo2)
    score2 = nli_entailment_score(lo2, lo1)
    return max(score1, score2)

def compute_nli_score(lo, generated_lo):
    result = nli_model(
        sequences=generated_lo,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="This example is {} with respect to: " + lo
    )
    for label, score in zip(result["labels"], result["scores"]):
        if label.lower() in ["entailment", "neutral"]:
            return score
    return 0.0

def hybrid_LO_evaluation(org_los, gen_los, module_name="Unknown Module"):
    results = []

    for org, gen in zip(org_los, gen_los):
        # Cosine similarity
        org_emb = embedder.encode(org, convert_to_tensor=True)
        gen_emb = embedder.encode(gen, convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(org_emb, gen_emb).item()

        # Gemini structure and coverage
        structure_feedback = gemini_similarity_and_structure(org, gen)
        structure_scores = extract_similarity_and_structure(structure_feedback)

        # Gemini final score
        final_feedback = gemini_final_rating(org, gen)
        final_score, final_reason = extract_final_rating(final_feedback)

        # NLI score
        nli_score = nli_entailment_score(org, gen)  # Score between 0–1

        # Normalized scores
        cosine_scaled = cosine_sim * 5
        nli_scaled = nli_score * 5

        # Updated weights (you can tune them)
        weights = {
            "cosine": 0.2,
            "nli": 0.2,
            "coverage": 0.3,
            "gemini": 0.3
        }

        hybrid_weighted_score = (
            weights["cosine"] * cosine_scaled +
            weights["nli"] * nli_scaled +
            weights["coverage"] * structure_scores['similarity_coverage_score'] +
            weights["gemini"] * final_score
        )

        # Feedbacks
        learner_feedback = generate_individual_lo_feedback(
            module_name=module_name,
            original_lo=org,
            gen_lo=gen,
            cosine_score=round(cosine_sim, 4),
            coverage_score=round(structure_scores['similarity_coverage_score'], 2),
            nli_score=round(nli_score, 4),
            bloom_match=structure_scores['bloom_taxonomy_match'],
            final_score=round(final_score, 2),
            gemini_reasoning=final_reason,
            gemini_structure_reason=structure_scores['reasoning'],
            hybrid_weighted_score=round(hybrid_weighted_score, 2),
            persona_type='learner'
        )

        instructor_feedback = generate_individual_lo_feedback(
            module_name=module_name,
            original_lo=org,
            gen_lo=gen,
            cosine_score=round(cosine_sim, 4),
            coverage_score=round(structure_scores['similarity_coverage_score'], 2),
            nli_score=round(nli_score, 4),
            bloom_match=structure_scores['bloom_taxonomy_match'],
            final_score=round(final_score, 2),
            gemini_reasoning=final_reason,
            gemini_structure_reason=structure_scores['reasoning'],
            hybrid_weighted_score=round(hybrid_weighted_score, 2),
            persona_type='instructor'
        )

        result = {
            "Original LO": org,
            "Generated LO": gen,
            "Cosine Similarity": round(cosine_sim, 4),
            "NLI Score": round(nli_score, 4),
            "Structure Coverage Score": round(structure_scores['similarity_coverage_score'], 2),
            "Gemini Score": round(final_score, 2),
            "Final Hybrid Score": round(hybrid_weighted_score, 2),
            "Learner Feedback": learner_feedback,
            "Instructor Feedback": instructor_feedback
        }

        results.append(result)

    return results

def generate_module_feedback(module_name, avg_cosine, avg_nli, avg_coverage, avg_gemini, avg_hybrid_score, persona_type: str):
    if persona_type == 'learner':
        user_prompt_template = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

As a learner, reflect on your module experience, focusing on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Semantic Closeness (Score Range: 0-5):**
    * **Definition:** This metric assesses how well the meaning of the original learning objectives is preserved in the generated content.
    * **Scoring:** 0 (significant divergence in meaning) to 5 (perfect preservation of meaning).

* **Logical Alignment (Score Range: 0-5):**
    * **Definition:** Measures how strongly the generated content is inferred or entailed by the original objectives. Higher values indicate better logical agreement.
    * **Scoring:** 0 (no logical connection) to 5 (strong logical inference).

* **Content Coverage (Score Range: 1-5):**
    * **Definition:** Checks if important concepts from the original are included and adequately addressed.
    * **Scoring:** 1 (missing key concepts) to 5 (comprehensive coverage).

* **AI Judgment (Score Range: 1-5):**
    * **Definition:** An AI-based assessment of the overall quality, tone, and accuracy of the learning materials based on the objectives.
    * **Scoring:** 1 (poor quality, inaccurate) to 5 (outstanding quality, highly accurate).

* **Final Module Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the overall aggregated score for the module's effectiveness, reflecting how well all elements coalesced for my learning experience.
    * **Scoring:** 1 (significant impediment to effective learning) to 5 (an outstanding and highly effective learning experience).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Semantic Closeness: {avg_cosine}/5
- Logical Alignment: {avg_nli}/5
- Content Coverage: {avg_coverage}/5
- AI Judgment: {avg_gemini}/5
- Final Rating: {avg_hybrid_score}/5

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, write a short learner-style reflection on the module's quality. Simulate how a learner, experienced engaging with the module's content and structure. The last sentence should clearly imply the overall experience of the learner.
"""
        response = model.generate_content(user_prompt_template)
        return response.text.strip()
    elif persona_type == 'instructor':
        instructor_prompt_template = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

As a pedagogical consultant, you are tasked with providing feedback on a module's syllabus to validate its design and effectiveness. This assessment aims to pinpoint design strengths and identify actionable areas for improvement.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Semantic Closeness (Score Range: 0-5):**
    * **Definition:** This metric quantifies how precisely the meaning of the original learning objectives is preserved in the generated content, ensuring what is promised is clearly and fully delivered.
    * **Scoring:** 0 (significant divergence in meaning) to 5 (perfect congruence; all objectives are fully supported and optimally aligned).

* **Logical Alignment (Score Range: 0-5):**
    * **Definition:** This metric rigorously measures how strongly the generated content is inferred or entailed by the original objectives, confirming the module's meaningful contribution.
    * **Scoring:** 0 (no logical connection to original objectives) to 5 (module content seamlessly integrates and significantly contributes to the overall learning flow).

* **Content Coverage (Score Range: 1-5):**
    * **Definition:** This metric evaluates the breadth, variety, and comprehensiveness of important concepts addressed within the module, ensuring appropriate intellectual rigor for the expected academic level.
    * **Scoring:** 1 (content is overly simplistic or lacks intellectual rigor for the expected level) to 5 (optimally stimulating, intellectually rigorous, and appropriately paced for advanced learning, with comprehensive content).

* **AI Judgment (Score Range: 1-5):**
    * **Definition:** This metric quantifies an AI's overall assessment of the module's instructional design excellence, reflecting its internal consistency, fidelity to learning outcomes, and strategic contribution.
    * **Scoring:** 1 (poor design; indicative of fundamental pedagogical flaws requiring urgent redesign) to 5 (excellent design; a model of instructional excellence, demonstrating outstanding clarity, coherence, and effectiveness).

* **Final Module Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the holistic and quantifiable assessment of the module's overall instructional design excellence, reflecting its internal consistency, fidelity to learning outcomes, and strategic contribution to the course.
    * **Scoring:** 1 (poor design; indicative of fundamental pedagogical flaws requiring urgent redesign) to 5 (excellent design; a model of instructional excellence, demonstrating outstanding clarity, coherence, and effectiveness).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Semantic Closeness: {avg_cosine}/5
- Logical Alignment: {avg_nli}/5
- Content Coverage: {avg_coverage}/5
- AI Judgment: {avg_gemini}/5
- Final Rating: {avg_hybrid_score}/5

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, deliver a precise pedagogical assessment and concrete, actionable recommendations for module enhancement for module "{module_name}". Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""
        response = model.generate_content(instructor_prompt_template)
        return response.text.strip()
    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")


def generate_course_feedback(course_name, avg_cosine, avg_nli, avg_coverage, avg_gemini, avg_hybrid_score, persona_type: str):
    if persona_type == 'learner':
        user_prompt_template = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

As a learner, reflect on your course experience, focusing on how effectively the course's interconnected elements (goals, content, intellectual demands) came together across all modules to deliver a cohesive, accessible, and ultimately enriching learning experience.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Semantic Closeness (Score Range: 0-5):**
    * **Definition:** This metric assesses how well the meaning of the original learning objectives is preserved throughout the course's content.
    * **Scoring:** 0 (significant divergence in meaning) to 5 (perfect preservation of meaning).

* **Logical Alignment (Score Range: 0-5):**
    * **Definition:** Measures how strongly the course content is inferred or entailed by the overall learning objectives. Higher values indicate better logical agreement.
    * **Scoring:** 0 (no logical connection) to 5 (strong logical inference).

* **Content Coverage (Score Range: 1-5):**
    * **Definition:** Checks if important concepts from the original course objectives are included and adequately addressed across all modules.
    * **Scoring:** 1 (missing key concepts) to 5 (comprehensive coverage).

* **AI Judgment (Score Range: 1-5):**
    * **Definition:** An AI-based assessment of the overall quality, tone, and accuracy of the course materials based on the objectives.
    * **Scoring:** 1 (poor quality, inaccurate) to 5 (outstanding quality, highly accurate).

* **Final Course Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the overall aggregated score for the course's effectiveness, reflecting how well all elements coalesced for my learning experience.
    * **Scoring:** 1 (significant impediment to effective learning) to 5 (an outstanding and highly effective learning experience).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Semantic Closeness: {avg_cosine}/5
- Logical Alignment: {avg_nli}/5
- Content Coverage: {avg_coverage}/5
- AI Judgment: {avg_gemini}/5
- Final Rating: {avg_hybrid_score}/5

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, write a short learner-style reflection on the course's quality. Simulate how a learner, experienced engaging with the course's content and structure. The last sentence should clearly imply the overall experience of the learner.
"""
        response = model.generate_content(user_prompt_template)
        return response.text.strip()
    elif persona_type == 'instructor':
        instructor_prompt_template = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a course's overall design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact.

As a pedagogical consultant, you are tasked with providing feedback on a course's syllabus to validate its design and effectiveness. This assessment aims to pinpoint design strengths and identify actionable areas for improvement for the entire course.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Semantic Closeness (Score Range: 0-5):**
    * **Definition:** This metric quantifies how precisely the meaning of the original learning objectives is preserved across all modules, ensuring consistent delivery of what is promised.
    * **Scoring:** 0 (significant divergence in meaning) to 5 (perfect congruence; all objectives are fully supported and optimally aligned throughout the course).

* **Logical Alignment (Score Range: 0-5):**
    * **Definition:** This metric rigorously measures how strongly the entire course content is inferred or entailed by the overarching learning objectives, confirming the course's cohesive and meaningful contribution.
    * **Scoring:** 0 (no logical connection across modules) to 5 (course content seamlessly integrates and significantly contributes to the overall learning flow).

* **Content Coverage (Score Range: 1-5):**
    * **Definition:** This metric evaluates the breadth, variety, and comprehensiveness of important concepts addressed throughout the course, ensuring appropriate intellectual rigor and complete topic exploration.
    * **Scoring:** 1 (content is overly simplistic or lacks intellectual rigor for the expected level) to 5 (optimally stimulating, intellectually rigorous, and appropriately paced for advanced learning, with comprehensive content).

* **AI Judgment (Score Range: 1-5):**
    * **Definition:** This metric quantifies an AI's overall assessment of the course's instructional design excellence, reflecting its internal consistency, fidelity to learning outcomes, and strategic contribution across all modules.
    * **Scoring:** 1 (poor design; indicative of fundamental pedagogical flaws requiring urgent redesign) to 5 (excellent design; a model of instructional excellence, demonstrating outstanding clarity, coherence, and effectiveness).

* **Final Course Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the holistic and quantifiable assessment of the course's overall instructional design excellence, reflecting its internal consistency, fidelity to learning outcomes, and strategic contribution to the course.
    * **Scoring:** 1 (poor design; indicative of fundamental pedagogical flaws requiring urgent redesign) to 5 (excellent design; a model of instructional excellence, demonstrating outstanding clarity, coherence, and effectiveness).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Semantic Closeness: {avg_cosine}/5
- Logical Alignment: {avg_nli}/5
- Content Coverage: {avg_coverage}/5
- AI Judgment: {avg_gemini}/5
- Final Rating: {avg_hybrid_score}/5

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, deliver a precise pedagogical assessment and concrete, actionable recommendations for course enhancement for course "{course_name}". Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        response = model.generate_content(instructor_prompt_template)
        return response.text.strip()
    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")


def process_course(course_path, metadata_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    metadata = load_metadata(metadata_path)

    course_name = metadata.get('Course', '')
    about_text = metadata.get('About this Course', '')
    module_keys = [key for key in metadata.keys() if key.startswith('Module ')]

    results = {}
    all_module_avg_hybrid_scores = []
    all_module_avg_cosine_scores = []
    all_module_avg_nli_scores = []
    all_module_avg_coverage_scores = []
    all_module_avg_gemini_scores = []

    for key in module_keys:
        module = metadata[key]
        module_name = module.get('Name', '')
        original_los = module.get('Learning Objectives', [])

        module_results = []
        module_lo_hybrid_scores = []
        module_lo_cosine_scores = []
        module_lo_nli_scores = []
        module_lo_coverage_scores = []
        module_lo_gemini_scores = []

        for lo in original_los:
            # Step 1: Extract concepts
            concepts = extract_concepts(course_name, about_text, module_name, lo)
            if not concepts:
                continue

            # Step 2: Assign concepts + Bloom level
            syllabus_for_lo = [lo]
            assignment = assign_concepts_and_bloom(syllabus_for_lo, concepts)
            if not assignment:
                continue

            # Step 3: Generate a single LO
            generated_lo_data = generate_learning_objectives(assignment)
            if not generated_lo_data or not generated_lo_data[0].get("Learning Objective"):
                continue

            generated_lo = generated_lo_data[0]["Learning Objective"]

            # Step 4: Compute embedding-based similarity
            orig_emb = embedder.encode(lo, convert_to_tensor=True)
            gen_emb = embedder.encode(generated_lo, convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(orig_emb, gen_emb).item()

            # Step 5: NLI entailment
            nli_score = compute_nli_score(lo, generated_lo)

            # Step 6: Gemini structure + Bloom match
            structure_feedback = gemini_similarity_and_structure(lo, generated_lo)
            structure_scores = extract_similarity_and_structure(structure_feedback)
            structure_reason = structure_scores.get("reasoning", "")
            structure_coverage_score = structure_scores.get("similarity_coverage_score", 0.0)
            bloom_match = structure_scores.get("bloom_taxonomy_match", "Unknown")

            # Step 7: Gemini generic eval
            final_feedback = gemini_final_rating(lo, generated_lo)
            final_score, final_reason = extract_final_rating(final_feedback)

            # Step 8: Hybrid Score (manual combination with default internal weights)
            hybrid_score = (
                0.1 * (cosine_sim * 5) +
                0.3 * (nli_score * 5) +
                0.3 * structure_coverage_score +
                0.3 * final_score
            )

            # Step 9: Feedback generation (Individual LO level)
            learner_feedback = generate_individual_lo_feedback(
                module_name,
                lo,
                generated_lo,
                cosine_sim,
                structure_coverage_score,
                nli_score,
                bloom_match,
                final_score,
                final_reason,
                structure_reason,
                hybrid_score,
                persona_type='learner'
            )

            instructor_feedback = generate_individual_lo_feedback(
                module_name,
                lo,
                generated_lo,
                cosine_sim,
                structure_coverage_score,
                nli_score,
                bloom_match,
                final_score,
                final_reason,
                structure_reason,
                hybrid_score,
                persona_type='instructor'
            )

            # Step 10: Store individual LO result
            lo_result = {
                "Original LO": lo,
                "Generated LO": generated_lo,
                "Cosine Similarity": round(cosine_sim, 4),
                "NLI Score": round(nli_score, 4),
                "Structure Coverage Score": round(structure_coverage_score, 2),
                "Gemini Score": round(final_score, 2),
                "Final Hybrid Score": round(hybrid_score, 2),
                "Learner Feedback": learner_feedback,
                "Instructor Feedback": instructor_feedback
            }
            module_results.append(lo_result)

            module_lo_hybrid_scores.append(hybrid_score)
            module_lo_cosine_scores.append(cosine_sim * 5) # Scale to 5 for consistency
            module_lo_nli_scores.append(nli_score * 5)     # Scale to 5 for consistency
            module_lo_coverage_scores.append(structure_coverage_score)
            module_lo_gemini_scores.append(final_score)

        if module_results:
            # Calculate module-wise averages
            avg_module_hybrid_score = np.mean(module_lo_hybrid_scores) if module_lo_hybrid_scores else 0.0
            avg_module_cosine_score = np.mean(module_lo_cosine_scores) if module_lo_cosine_scores else 0.0
            avg_module_nli_score = np.mean(module_lo_nli_scores) if module_lo_nli_scores else 0.0
            avg_module_coverage_score = np.mean(module_lo_coverage_scores) if module_lo_coverage_scores else 0.0
            avg_module_gemini_score = np.mean(module_lo_gemini_scores) if module_lo_gemini_scores else 0.0

            # Generate module-wise feedback
            module_learner_feedback = generate_module_feedback(
                module_name,
                round(avg_module_cosine_score, 2),
                round(avg_module_nli_score, 2),
                round(avg_module_coverage_score, 2),
                round(avg_module_gemini_score, 2),
                round(avg_module_hybrid_score, 2),
                persona_type='learner'
            )
            module_instructor_feedback = generate_module_feedback(
                module_name,
                round(avg_module_cosine_score, 2),
                round(avg_module_nli_score, 2),
                round(avg_module_coverage_score, 2),
                round(avg_module_gemini_score, 2),
                round(avg_module_hybrid_score, 2),
                persona_type='instructor'
            )

            results[key] = {
                "Module Name": module_name,
                "Learning Objectives": module_results,
                "Module Averages": {
                    "Avg Final Hybrid Score": round(avg_module_hybrid_score, 2),
                    "Avg Cosine Similarity": round(avg_module_cosine_score, 2),
                    "Avg NLI Score": round(avg_module_nli_score, 2),
                    "Avg Structure Coverage Score": round(avg_module_coverage_score, 2),
                    "Avg Gemini Score": round(avg_module_gemini_score, 2)
                },
                "Module Learner Feedback": module_learner_feedback,
                "Module Instructor Feedback": module_instructor_feedback
            }
            all_module_avg_hybrid_scores.append(avg_module_hybrid_score)
            all_module_avg_cosine_scores.append(avg_module_cosine_score)
            all_module_avg_nli_scores.append(avg_module_nli_score)
            all_module_avg_coverage_scores.append(avg_module_coverage_score)
            all_module_avg_gemini_scores.append(avg_module_gemini_score)
        else:
            print(f"⚠️ Skipping missing module or no valid LOs: {key}")

    # Calculate course-wise averages
    avg_course_hybrid_score = np.mean(all_module_avg_hybrid_scores) if all_module_avg_hybrid_scores else 0.0
    avg_course_cosine_score = np.mean(all_module_avg_cosine_scores) if all_module_avg_cosine_scores else 0.0
    avg_course_nli_score = np.mean(all_module_avg_nli_scores) if all_module_avg_nli_scores else 0.0
    avg_course_coverage_score = np.mean(all_module_avg_coverage_scores) if all_module_avg_coverage_scores else 0.0
    avg_course_gemini_score = np.mean(all_module_avg_gemini_scores) if all_module_avg_gemini_scores else 0.0


    # Generate course-wise feedback
    course_learner_feedback = generate_course_feedback(
        course_name,
        round(avg_course_cosine_score, 2),
        round(avg_course_nli_score, 2),
        round(avg_course_coverage_score, 2),
        round(avg_course_gemini_score, 2),
        round(avg_course_hybrid_score, 2),
        persona_type='learner'
    )
    course_instructor_feedback = generate_course_feedback(
        course_name,
        round(avg_course_cosine_score, 2),
        round(avg_course_nli_score, 2),
        round(avg_course_coverage_score, 2),
        round(avg_course_gemini_score, 2),
        round(avg_course_hybrid_score, 2),
        persona_type='instructor'
    )


    final_output = {
        "Modules": results,
        "Course": course_name,
        "Course Averages": {
            "Avg Final Hybrid Score": round(avg_course_hybrid_score, 2),
            "Avg Cosine Similarity": round(avg_course_cosine_score, 2),
            "Avg NLI Score": round(avg_course_nli_score, 2),
            "Avg Structure Coverage Score": round(avg_course_coverage_score, 2),
            "Avg Gemini Score": round(avg_course_gemini_score, 2)
        },
        "Course Learner Feedback": course_learner_feedback,
        "Course Instructor Feedback": course_instructor_feedback,
    }

    course_folder_name = os.path.basename(os.path.normpath(course_path))
    output_file = os.path.join(
        output_path,
        f"{course_folder_name}_concept_objectives_weighted_feedback_nli_aggregated.json"
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved concept-based objectives to: {output_file}")

if __name__ == "__main__":
    process_course(
        course_path=BASE_PATH,
        metadata_path=METADATA_PATH,
        output_path=OUTPUT_PATH
    )