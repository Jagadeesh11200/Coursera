import os
import json
import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai

# Load NLI model once (at global scope)
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load API Key
load_dotenv('api_key.env')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-001')

# BASE_PATH setup
BASE_PATH = r""
METADATA_PATH = os.path.join(BASE_PATH, "metadata.json")
OUTPUT_PATH = r""
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load NLP tools
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def compute_cosine_similarity(prereqs_a_text, prereqs_b_text):
    if not prereqs_a_text or not prereqs_b_text:
        return 0.0

    a_embeddings = embedder.encode([prereqs_a_text], convert_to_tensor=True)
    b_embeddings = embedder.encode([prereqs_b_text], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(a_embeddings, b_embeddings).item()
    return round(cosine_sim, 4)


def parse_prerequisites(prereq_text):
    technical = []
    conceptual = []

    current_section = None
    for line in prereq_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("technical prerequisites"):
            current_section = "technical"
        elif line.lower().startswith("conceptual prerequisites"):
            current_section = "conceptual"
        elif line.startswith("- ") and current_section:
            item = line[2:].strip()
            if current_section == "technical":
                technical.append(item)
            elif current_section == "conceptual":
                conceptual.append(item)

    return {
        "Technical Prerequisites": technical,
        "Conceptual Prerequisites": conceptual
    }

def verify_course_prerequisites(course_description):
    prompt = f"""
You are an expert course designer specializing in rigorous and honest evaluation of course documentation. Your task is to review the following course description and determine with clarity whether it includes explicit or clearly implied prerequisites.

Guidelines:
- Only consider statements within the course description itself. Do NOT infer or assume requirements based on the course title or general academic norms.
- If the description does mention prerequisites, you must:
    1. Quote the relevant text verbatim.
    2. Paraphrase what the prerequisite is.
    3. Clearly indicate that prerequisites are explicitly present.
- If the prerequisites are implied but not clearly stated, you must:
    1. Explain how you interpreted the implication.
    2. Clearly state that the prerequisites are only implied, not explicit.
- If the course description does NOT contain any prerequisites or implications within the text, simply respond: "No Prerequisites found."
- If there is any ambiguity or missing information, honestly acknowledge it and do NOT assume any prerequisites.
- Do NOT draw on outside knowledge or common sense. Focus strictly on the provided course description.
- Do NOT fill gaps with your own reasoning. Always be explicit if information is absent or insufficient.
- Do NOT make up any content.

Course Description: {course_description}
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_prerequisites_prompt(module_name: str, syllabus_points: list, learning_objectives: list) -> str:
    return f"""
You are an instructional designer.

Given the following:
- **Module Title**: {module_name}
- **Syllabus Points**:
{json.dumps(syllabus_points, indent=2)}

- **Learning Objectives**:
{json.dumps(learning_objectives, indent=2)}

Identify what a learner needs to already know to fully understand this module.

Output format:
Technical Prerequisites:
- ...

Conceptual Prerequisites:
- ...

Guidelines:
- "Technical Prerequisites" should include tools, libraries, coding knowledge, or frameworks required.
- "Conceptual Prerequisites" should include key theories, mathematical foundations, or domain-specific concepts needed.
- Keep the list specific and avoid vague or overly broad items like "math".
- Only return the bullet-pointed output.
"""

def generate_module_prerequisites(module_name, syllabus, learning_objectives):
    prompt = generate_prerequisites_prompt(module_name, syllabus, learning_objectives)
    response = model.generate_content(prompt)
    return response.text.strip()

def compute_nli_entailment_score(premise, hypothesis, nli_pipeline):
    try:
        result = nli_pipeline(hypothesis, candidate_labels=[premise], multi_label=False)
        if isinstance(result, list):
            score = result[0]['scores'][0]
        else:
            score = result['scores'][0]
        return round(score, 4)
    except Exception as e:
        print(f"NLI error: {e}")
        return 0.0

def compute_nli_score(prereqs_course, prereqs_module, nli_pipeline):
    def flatten(pr):
        return pr.get("Technical Prerequisites", []) + pr.get("Conceptual Prerequisites", [])

    course_items = flatten(prereqs_course)
    module_items = flatten(prereqs_module)
    if not course_items or not module_items:
        return 0.0

    scores = []
    for mod in module_items:
        max_score = max([compute_nli_entailment_score(course, mod, nli_pipeline) for course in course_items])
        scores.append(max_score)
    return round(sum(scores)/len(scores), 4) if scores else 0.0

def gemini_prerequisite_structure_score(module_prereq, course_prereq):
    prompt = f"""
You are an instructional designer evaluating prerequisite alignment.

Compare the **Module-Level Prerequisites** with the **Course-Level Prerequisites**:

Module-Level Prerequisites:
{module_prereq}

Course-Level Prerequisites:
{course_prereq}

1. Assess whether the module prerequisites are well-aligned with the broader course-level expectations.
2. Identify if any critical technical or conceptual gaps are present.
3. Rate the coverage and alignment from 1.00 to 5.00.

Respond in this format:
StructureCoverageScore: <float between 1.00 to 5.00>
Reason: <brief justification>
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_structure_coverage_score(text):
    score = re.search(r"StructureCoverageScore:\s*([0-9]+\.[0-9]+)", text)
    reason = re.search(r"Reason:\s*(.*)", text, re.DOTALL)
    return float(score.group(1)) if score else 0.0, reason.group(1).strip() if reason else ""

def gemini_prerequisite_final_score(module_prereq, course_prereq):
    prompt = f"""
As a curriculum evaluator, rate the overall alignment of **module-level prerequisites** with the **course-level prerequisites**.

Module-Level Prerequisites:
{module_prereq}

Course-Level Prerequisites:
{course_prereq}

Consider:
- Relevance and completeness of technical and conceptual coverage
- Alignment with course difficulty expectations
- Absence of critical omissions

Respond in this format:
Rating: <float between 1.00 to 5.00>
Reason: <brief explanation>
"""
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_learner_feedback_prerequisites_hybrid(module_name, course_prereqs_text, module_prereqs_text, cosine_score, nli_score, final_score, gemini_reasoning, gemini_structure_reason, hybrid_weighted_score):
    step1 = f"""
You are evaluating a course module titled "{module_name}" to assist a prospective learner in understanding whether the **expected background knowledge** matches what they actually need to succeed in the module.

--- Course-Level Prerequisites ---
{course_prereqs_text}

--- Module-Level Prerequisites ---
{module_prereqs_text}

Evaluation Metrics (scale: 0–5):
- **Semantic Match**: {cosine_score} → Checks if the prerequisites described in the module match the meaning of those in the course. Higher = more similar ideas.
- **Logical Alignment**: {nli_score} → Measures how clearly the module’s expectations logically follow from the course prerequisites. Higher = better alignment.
- **AI Judgment**: {final_score} → A language model’s rating of how well the prerequisites are structured and communicated.
- **Final Match Score**: {hybrid_weighted_score} → Combines the above signals to give an overall match between module and course-level prerequisites.

--- Gemini Explanations ---
- Final Reasoning:
{gemini_reasoning}

- Structural Feedback:
{gemini_structure_reason}

Step 1: Based on the above data, assess whether the module prerequisites are well aligned with the broader course expectations. Was the background knowledge needed clearly communicated? Identify whether the alignment is strong, moderate, or weak, and explain what helped or hurt the alignment (e.g., vague concepts, missing tools, over-specific assumptions).
"""
    analysis = model.generate_content(step1).text.strip()

    step2 = f"""
Write a short 3–4 sentence summary **from a learner’s perspective** about whether the background knowledge needed for this module feels clear and appropriate.

Style Guidelines:
- Use the tone of a thoughtful learner sharing insights with peers (not overly casual or chatty)
- Use natural, relaxed phrasing — avoid expert tone or academic language
- Use passive voice where it fits naturally
- Do not use “I” or “we”
- Avoid phrases like “just checked”, “heads-up”, “it feels like”, or “seems to be”
- The comment should sound confident, helpful, and focused on whether the module looks manageable, too advanced, or well-aligned with earlier parts of the course

Write the comment based only on the analysis below:

{analysis}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback

def generate_instructor_feedback_prerequisites_hybrid(module_name, course_prereqs_text, module_prereqs_text, cosine_score, nli_score, final_score, gemini_reasoning, gemini_structure_reason, hybrid_weighted_score):
    step1 = f"""
You are reviewing prerequisite alignment for the course module titled "{module_name}".

--- Prerequisite Information ---

Course-Level Prerequisites:
{course_prereqs_text}

Module-Level Prerequisites:
{module_prereqs_text}

--- Evaluation Metrics (0 to 5) ---

- **Semantic Match**: {cosine_score}  
  Indicates how well the meaning of module prerequisites aligns with the course-level ones.

- **Logical Alignment**: {nli_score}  
  Reflects whether the module expectations logically follow from the course prerequisites.

- **AI-Based Judgment**: {final_score}  
  A language model’s overall evaluation of prerequisite quality and alignment.

- **Final Weighted Score**: {hybrid_weighted_score}  
  Aggregated score summarizing the alignment strength across all signals.

--- Gemini Model Explanations ---

- **Alignment Reasoning**:  
{gemini_reasoning}

- **Structural Feedback**:  
{gemini_structure_reason}

Step 1:  
Assess the prerequisite alignment using the metrics and explanations above. Focus on three key dimensions: (1) semantic consistency with course expectations, (2) completeness and clarity of skills/tools/concepts mentioned, and (3) potential mismatches or missing assumptions that could hinder learner readiness.

Write a concise evaluation in two parts:
1. Strengths observed in prerequisite clarity, alignment, or relevance.
2. Any identified gaps or issues, followed by actionable suggestions to improve the prerequisite design.
Limit to 3–4 sentences total.
"""
    analysis = model.generate_content(step1).text.strip()

    step2 = f"""
Write a feedback message for the instructor based on the analysis below.

Guidelines:
- Do NOT reference metric names or numbers directly.
- Focus on concrete instructional insights: strengths and needed improvements.
- Tone: Professional, respectful, and improvement-oriented.
- Do NOT include introductory statements.
- Length: Strictly 3–4 sentences.

Detailed Analysis:
{analysis}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback


def generate_course_feedback(course_name, avg_cosine, avg_nli, avg_coverage, avg_gemini, avg_hybrid_score, persona_type: str):
    if persona_type == 'learner':
        user_prompt_template = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

As a learner, reflect on your course experience, focusing on how well the course's **prerequisites** prepared you for the content that followed. Consider how clearly the expected prior knowledge was defined, how well it aligned with what was needed, and whether the transition into the course felt smooth and coherent.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Semantic Closeness (Score Range: 0–5):**
    * **Definition:** This metric assesses how well the meaning of the original prerequisite expectations was reflected in the course material.
    * **Scoring:** 0 (significant divergence in meaning) to 5 (perfect preservation of meaning).

* **Logical Alignment (Score Range: 0–5):**
    * **Definition:** Measures how strongly the course content logically followed from the stated prerequisites. Higher values indicate clearer instructional continuity.
    * **Scoring:** 0 (no logical connection) to 5 (strong logical progression).

* **Content Coverage (Score Range: 1–5):**
    * **Definition:** Checks if the prerequisite knowledge was properly reinforced or integrated before or during the course delivery.
    * **Scoring:** 1 (gaps in foundational knowledge) to 5 (comprehensively addressed and reinforced).

* **AI Judgment (Score Range: 1–5):**
    * **Definition:** An AI-based assessment of how effectively the prerequisites were framed and supported throughout the course.
    * **Scoring:** 1 (poorly addressed) to 5 (highly supportive and well-scaffolded).

* **Final Course Quality Rating (Overall Score Range: 1–5):**
    * **Definition:** This is the overall aggregated score for how well the prerequisites set the stage for effective learning.
    * **Scoring:** 1 (created confusion or learning obstacles) to 5 (enabled a smooth and effective learning journey).

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
* **Focus:** My perceived clarity, preparedness, and how the prerequisites impacted the ease of transitioning into course content.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, write a short learner-style reflection on how effectively the course prerequisites supported the learning experience. Simulate how a learner would perceive the preparedness and transition. The last sentence should clearly imply the overall experience of the learner.
"""
        response = model.generate_content(user_prompt_template)
        return response.text.strip()
    elif persona_type == 'instructor':
        instructor_prompt_template = f"""
    You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a course's prerequisite design and its instructional integration across all modules. Your insights aim to ensure that the course is accessible, well-sequenced, and effectively prepares learners for success.

    As a pedagogical consultant, you are tasked with reviewing how clearly and effectively the **prerequisites** are defined and how well they support a smooth instructional transition into the course content. Your goal is to validate whether the foundational expectations are logically sound, appropriately scoped, and pedagogically integrated.

    **Evaluation Metrics & Scoring Principles (for your understanding):**

    * **Semantic Closeness (Score Range: 0–5):**
        * **Definition:** Assesses how accurately the intended meaning of the course prerequisites is preserved across the content and instructional flow.
        * **Scoring:** 0 (significant mismatch between expected and actual prerequisite coverage) to 5 (clear, consistent prerequisite scaffolding throughout).

    * **Logical Alignment (Score Range: 0–5):**
        * **Definition:** Evaluates how logically the course content builds upon the stated prerequisites, ensuring a coherent entry point into the material.
        * **Scoring:** 0 (no logical relationship) to 5 (clear progression from prerequisite knowledge into core learning).

    * **Content Coverage (Score Range: 1–5):**
        * **Definition:** Examines the degree to which prerequisite knowledge is reinforced or revisited during course delivery, ensuring foundational gaps are bridged.
        * **Scoring:** 1 (critical prerequisite gaps unaddressed) to 5 (thoroughly scaffolded and supported).

    * **AI Judgment (Score Range: 1–5):**
        * **Definition:** Captures an AI’s comprehensive evaluation of how well prerequisite design supports learner preparedness, instructional pacing, and accessibility.
        * **Scoring:** 1 (poorly structured, misaligned prerequisites) to 5 (exemplary prerequisite integration and learner onboarding).

    * **Final Course Quality Rating (Overall Score Range: 1–5):**
        * **Definition:** A holistic rating of how effectively the course’s prerequisite structure supports learner readiness and smooth content progression.
        * **Scoring:** 1 (prerequisites are a barrier to learning) to 5 (prerequisites are enabling, well-aligned, and pedagogically sound).

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
    * **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to how prerequisites are framed, integrated, or reinforced in the course.
    * **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

    **Prompt Instruction:**
    Given the underlying quality observations: Semantic Closeness: {avg_cosine}, Logical Alignment: {avg_nli}, Content Coverage: {avg_coverage}, AI Judgment: {avg_gemini}, and a Final Rating: {avg_hybrid_score}, deliver a precise pedagogical assessment and concrete, actionable recommendations for improving the prerequisite design and instructional integration for course "{course_name}". Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
    """
        response = model.generate_content(instructor_prompt_template)
        return response.text.strip()
    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")
    
def is_no_prerequisites(response_text):
    # Normalize whitespace, case, and punctuation
    normalized = re.sub(r'[^\w\s]', '', response_text).strip().lower()
    return normalized == "no prerequisites found"


def extract_final_prerequisite_score(text):
    match = re.search(r"Rating:\s*([0-9]+\.[0-9]+)", text)
    reason = re.search(r"Reason:\s*(.*)", text, re.DOTALL)
    return float(match.group(1)) if match else 0.0, reason.group(1).strip() if reason else ""

def process_course_prerequisites(course_path, metadata_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    metadata = load_metadata(metadata_path)

    course_name = metadata.get("Course", "")
    course_description = metadata.get("About this Course", "")  # Make sure this field exists

    # Run prerequisite presence check
    prereq_verification = verify_course_prerequisites(course_description)
    print(prereq_verification)
    if is_no_prerequisites(prereq_verification):
        print("🛑 No Prerequisites found.")

        output = {
            "Course": course_name,
            "Course Level Evaluation": {
                "Course Hybrid Score": "N/A",
                "Learner Perspective Assessment": "N/A",
                "Instructor Feedback": "N/A"
            }
        }

        course_folder_name = os.path.basename(os.path.normpath(course_path))
        output_file = os.path.join(
            output_path,
            f"{course_folder_name}_prerequisite_similarity_weighted_feedback new.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved empty prerequisite evaluation to: {output_file}")
        return  # End early
    else:
        print("✅ Prerequisites are Present.")

    # Extract structured prerequisites and convert to formatted text
    prereq_data = metadata.get("Prerequisites", {})
    course_prereqs_text = "Technical Prerequisites:\n" + "\n".join(
        f"- {item}" for item in prereq_data.get("Technical Prerequisites", [])
    ) + "\n\nConceptual Prerequisites:\n" + "\n".join(
        f"- {item}" for item in prereq_data.get("Conceptual Prerequisites", [])
    )
    course_prereqs = parse_prerequisites(course_prereqs_text)
    course_name = metadata.get("Course", "")
    module_keys = [key for key in metadata if key.startswith("Module ")]

    results = {}
    cosine_scores, nli_scores, coverage_scores, gemini_scores, hybrid_scores = [], [], [], [], []

    for key in module_keys:
        module = metadata[key]
        module_name = module.get("Name", "")
        syllabus = module.get("Syllabus", [])
        learning_objectives = module.get("Learning Objectives", [])

        module_prereqs_text = generate_module_prerequisites(module_name, syllabus, learning_objectives)
        module_prereqs = parse_prerequisites(module_prereqs_text)

        if not module_prereqs.get("Technical Prerequisites") and not module_prereqs.get("Conceptual Prerequisites"):
            continue

        cosine_score = compute_cosine_similarity(course_prereqs_text, module_prereqs_text)
        cosine_scaled = cosine_score * 5

        nli_score = compute_nli_score(course_prereqs, module_prereqs, nli_model)
        nli_scaled = nli_score * 5

        structure_feedback = gemini_prerequisite_structure_score(module_prereqs_text, course_prereqs_text)
        coverage_score, structure_reason = extract_structure_coverage_score(structure_feedback)

        final_feedback = gemini_prerequisite_final_score(module_prereqs_text, course_prereqs_text)
        final_score, final_reason = extract_final_prerequisite_score(final_feedback)

        weights = {"cosine": 0.1, "nli": 0.3, "coverage": 0.3, "gemini": 0.3}
        hybrid_score = (
            weights["cosine"] * cosine_scaled +
            weights["nli"] * nli_scaled +
            weights["coverage"] * coverage_score +
            weights["gemini"] * final_score
        )

        # Store individual scores for course-level aggregation
        cosine_scores.append(cosine_scaled)
        nli_scores.append(nli_scaled)
        coverage_scores.append(coverage_score)
        gemini_scores.append(final_score)
        hybrid_scores.append(hybrid_score)

        learner_feedback = generate_learner_feedback_prerequisites_hybrid(
            module_name, course_prereqs_text, module_prereqs_text,
            cosine_score, nli_score, final_score, final_reason, structure_reason, hybrid_score
        )

        instructor_feedback = generate_instructor_feedback_prerequisites_hybrid(
            module_name, course_prereqs_text, module_prereqs_text,
            cosine_score, nli_score, final_score, final_reason, structure_reason, hybrid_score
        )

        results[key] = {
            "Module Name": module_name,
            "Extracted Prerequisites": module_prereqs,
            "Cosine Similarity": round(cosine_score, 4),
            "NLI Score": round(nli_score, 4),
            "Structure Coverage Score": round(coverage_score, 2),
            "Gemini Final Score": round(final_score, 2),
            "Final Hybrid Score": round(hybrid_score, 2),
            "Learner Perspective Assessment": learner_feedback,
            "Instructor Feedback": instructor_feedback
        }

    # Compute course-level averages
    avg_cosine = round(sum(cosine_scores) / len(cosine_scores), 2) if cosine_scores else 0
    avg_nli = round(sum(nli_scores) / len(nli_scores), 2) if nli_scores else 0
    avg_coverage = round(sum(coverage_scores) / len(coverage_scores), 2) if coverage_scores else 0
    avg_gemini = round(sum(gemini_scores) / len(gemini_scores), 2) if gemini_scores else 0
    avg_hybrid_score = round(sum(hybrid_scores) / len(hybrid_scores), 2) if hybrid_scores else 0

    # Generate course-level feedback using the new prompt
    learner_course_feedback = generate_course_feedback(
        course_name, avg_cosine, avg_nli, avg_coverage, avg_gemini, avg_hybrid_score, persona_type="learner"
    )
    instructor_course_feedback = generate_course_feedback(
        course_name, avg_cosine, avg_nli, avg_coverage, avg_gemini, avg_hybrid_score, persona_type="instructor"
    )

    final_output = {
        "Course": course_name,
        "Course Prerequisites": course_prereqs_text,
        "Course-Level Scores": {
            "Average Semantic Closeness": avg_cosine,
            "Average Logical Alignment": avg_nli,
            "Average Coverage": avg_coverage,
            "Average AI Judgment": avg_gemini,
            "Final Hybrid Score": avg_hybrid_score
        },
        "Course-Level Evalutation": {
            "Learner Perspective": learner_course_feedback,
            "Instructor Perspective": instructor_course_feedback
        },
        "Modules": results
    }

    course_folder_name = os.path.basename(os.path.normpath(course_path))
    output_file = os.path.join(
        output_path,
        f"{course_folder_name}_prerequisite_similarity_weighted_feedback new.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved prerequisite similarity results to: {output_file}")

if __name__ == "__main__":
    process_course_prerequisites(
        course_path=BASE_PATH,
        metadata_path=METADATA_PATH,
        output_path=OUTPUT_PATH
    )