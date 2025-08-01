# Combined LOCS v1.5, AOAS, and RDI v2 Evaluator

import os
import re
import json
import math
import fitz
import numpy as np
import spacy
from tqdm import tqdm
from functools import lru_cache
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# === Initialization ===
load_dotenv('api_key.env')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class CourseEvaluator:
    def __init__(self, course_folder: str):
        self.course_folder = course_folder
        self.chunk_size = 100
        self.threshold = 0.5
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.metadata = self.load_metadata()
        self.modules = self.extract_modules()
        self.nlp = spacy.load("en_core_web_sm")
        self.bloom_verbs = {
            'remember': ['list', 'recall', 'define', 'identify', 'name', 'state', 'describe'],
            'understand': ['explain', 'interpret', 'summarize', 'classify', 'compare', 'discuss'],
            'apply': ['solve', 'demonstrate', 'calculate', 'implement', 'execute', 'use'],
            'analyze': ['analyze', 'examine', 'differentiate', 'organize', 'attribute', 'deconstruct'],
            'evaluate': ['evaluate', 'critique', 'judge', 'assess', 'defend', 'support'],
            'create': ['create', 'design', 'generate', 'plan', 'produce', 'construct']
        }

    def load_metadata(self):
        path = os.path.join(self.course_folder, 'metadata.json')
        with open(path, 'r') as f:
            return json.load(f)

    def extract_modules(self):
        return [
            {"Module Name": mod['Name'], "Learning Objectives": mod['Learning Objectives']}
            for key, mod in self.metadata.items() if key.startswith("Module")
        ]

    def extract_text(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_pdf_text(self, filepath):
        text = []
        with fitz.open(filepath) as doc:
            for page in doc:
                text.append(page.get_text())
        return ' '.join(text)

    def chunk_text(self, text):
        words = re.findall(r'\w+', text)
        return [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    @lru_cache(maxsize=256)
    def gemini_summarise(self, text: str, model_name="models/gemini-2.0-flash",max_tokens=512) -> str:
        prompt = (
            "You are a helpful teaching assistant.\n"
            "Summarise the following reading in 10-12 short bullet points, focusing on the key concepts or skills it covers.\n"
            "Do **NOT** add any extra commentary.\n\n"
            "---- BEGIN DOCUMENT ----\n"
            f"{text[:20000]}\n"
            "---- END DOCUMENT ----"
        )
        gm = genai.GenerativeModel(model_name)
        resp = gm.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        return resp.text

    def extract_action_verbs(self, text):
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if token.pos_ in {"VERB", "AUX"}]

    def classify_cognitive_level(self, text):
        verbs = self.extract_action_verbs(text)
        level_scores = {level: 0 for level in self.bloom_verbs}
        for verb in verbs:
            for level, level_verbs in self.bloom_verbs.items():
                if verb in level_verbs:
                    level_scores[level] += 1
        total = sum(level_scores.values())
        return {k: v / total for k, v in level_scores.items()} if total > 0 else level_scores

    def js_similarity(self, p1, p2):
        a = np.array([p1.get(k, 0) for k in self.bloom_verbs])
        b = np.array([p2.get(k, 0) for k in self.bloom_verbs])
        a, b = a + 1e-10, b + 1e-10
        a, b = a / np.sum(a), b / np.sum(b)
        m = 0.5 * (a + b)
        js = 0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m))
        return 1 - js / np.log(2)

    def compute_lops(self, learning_objectives, resources):
        total, covered = len(learning_objectives), 0
        for lo in learning_objectives:
            lo_emb = self.model.encode(lo, convert_to_tensor=True)
            covered_types = []
            for r_type, texts in resources.items():
                if not texts: continue
                chunks = [c for t in texts for c in self.chunk_text(t)]
                if not chunks: continue
                embs = self.model.encode(chunks, convert_to_tensor=True, batch_size=32)
                sims = util.cos_sim(lo_emb, embs)
                if np.max(sims.cpu().numpy()) >= self.threshold:
                    covered_types.append(r_type)
            if len(covered_types) >= 2:
                covered += 1
        score = round((covered / total) * 5, 2) if total > 0 else 0.0
        return score

    def compute_aoas(self, learning_objectives: List[str], assessments: List[Tuple[str, str]]) -> float:
        sem_w, act_w = 0.7, 0.3
        scaled_scores = []
        all_chunks = [chunk for _, text in assessments for chunk in self.chunk_text(text)]
        if not all_chunks:
            return 0.0
        assessment_embs = self.model.encode(all_chunks, convert_to_tensor=True)
        assessment_profile = self.classify_cognitive_level(' '.join(t for _, t in assessments))
        for lo in learning_objectives:
            lo_emb = self.model.encode(lo, convert_to_tensor=True)
            max_sim = np.max(util.cos_sim(lo_emb, assessment_embs).cpu().numpy())
            lo_profile = self.classify_cognitive_level(lo)
            act_sim = self.js_similarity(lo_profile, assessment_profile)
            combined = (sem_w * max_sim + act_w * act_sim) * 5
            scaled_scores.append(combined)
        return round(np.mean(scaled_scores), 2) if scaled_scores else 0.0

    def compute_rdi(self, qualified: Dict[str, bool]) -> float:
        types = ["videos", "quizzes", "readings", "labs"]
        present = sum(qualified[t] for t in types if t in qualified)
        if present <= 1: return 0.0
        probs = [1/present] * present
        entropy = -sum(p * math.log2(p) for p in probs)
        return round((entropy / math.log2(4)) * 5, 2)
    
    def gemini_generate(self, prompt: str, model_name="models/gemini-2.0-flash",max_tokens: int = 512) -> str:
        gm = genai.GenerativeModel(model_name)
        resp = gm.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        return resp.text if hasattr(resp, "text") else str(resp)
    
    def learner_lops_prompt(self, course_name, lops_score):
        return f"""
    You are advising a learner who is deciding whether to take the course "{course_name}".
    The internal Learning Objective Penetration Score (LOPS) for this course is {lops_score} out of 5.

    Step 1: Based on this score, reflect on how well the course materials (videos, readings, quizzes) support the stated learning goals.

    Step 2: Consider whether there’s a balanced use of resources that truly help the learner understand key concepts.

    Step 3: Write a clear and concise explanation (2 to 4 sentences) that helps the learner decide if this course teaches what it promises. Avoid using the score in the response. Keep language non-technical.
    """

    def learner_rdi_prompt(self, course_name, rdi_score):
        return f"""
    You are advising a learner who is deciding whether to take the course "{course_name}".
    The internal Resource Diversity Index (RDI) score for this course is {rdi_score} out of 5.

    Step 1: Use this score to assess whether the course includes a good variety of materials (videos, readings, quizzes, labs).

    Step 2: Consider if this mix can help different types of learners and keep them engaged.

    Step 3: Now write a short explanation (2 to 4 sentences) for the learner. Don’t include the score. Use simple, helpful language.
    """

    def learner_aoas_prompt(self, course_name, aoas_score):
        return f"""
    You are advising a learner who is deciding whether to take the course "{course_name}".
    The internal Assessment-Objective Alignment Score (AOAS) for this course is {aoas_score} out of 5.

    Step 1: Consider how well the course assessments (like quizzes or labs) help learners practice and reinforce what’s being taught.

    Step 2: Based on the score, think about whether these assessments feel meaningful and well-aligned with the course content.

    Step 3: Write a short, friendly explanation (2 to 4 sentences) to help the learner understand if the course includes useful and relevant practice. Avoid mentioning the score.
    """


    def instructor_lops_prompt(self, course_name, lops_score):
        return f"""
    You are reviewing the course "{course_name}" and providing feedback to the instructor.

    **Learning Objective Penetration Score (LOPS): {lops_score}/5**

    Step 1: Analyze how well the course materials (videos, quizzes, readings, etc.) support the stated learning objectives.
    Step 2: Identify any gaps — for example, objectives mentioned but not supported by multiple resource types.
    Step 3: Offer detailed and practical suggestions to improve alignment between objectives and resources. Mention the importance of reinforcing key concepts through diverse modalities.
    """

    def instructor_rdi_prompt(self, course_name, rdi_score):
        return f"""
    You are reviewing the course "{course_name}" and providing feedback to the instructor.

    **Resource Diversity Index (RDI): {rdi_score}/5**

    Step 1: Evaluate whether the course includes a balanced mix of learning formats — such as video lectures, readings, interactive quizzes, and labs or walkthroughs.
    Step 2: Reflect on whether this mix caters to different learning styles (e.g., visual, reflective, practical).
    Step 3: Provide specific and helpful recommendations. If the score is low, suggest integrating underused formats or improving transitions between them to enhance learner engagement.
    """

    def instructor_aoas_prompt(self, course_name, aoas_score):
        return f"""
    You are reviewing the course "{course_name}" and providing feedback to the instructor.

    **Assessment-Objective Alignment Score (AOAS): {aoas_score}/5**

    Step 1: Examine whether the assessments (quizzes, labs, exercises) are clearly aligned with the course's learning objectives.
    Step 2: Identify any disconnects — such as assessments that miss core objectives or emphasize peripheral content.
    Step 3: Give thoughtful and practical feedback. Recommend ways to ensure assessments reinforce what is taught, and encourage active learning. Highlight opportunities to include more formative or summative checks where needed.
    """

    def run(self):
        results = {"Modules": []}
        total_lops, total_aoas, total_objectives, rdi_scores = 0, 0, 0, []

        for module in tqdm(self.modules, desc="Evaluating Modules"):
            mod_name = module["Module Name"]
            week_match = re.search(r"Week\s*(\d+)", mod_name)
            if not week_match: continue
            week_folder = os.path.join(self.course_folder, f"Week {week_match.group(1)}")
            if not os.path.exists(week_folder): continue

            subtitles, quizzes, readings, labs = [], [], [], []
            subtitle_dirs = set()

            for root, _, files in os.walk(week_folder):
                for file in files:
                    fpath = os.path.join(root, file)
                    if file.lower() == 'subtitle.txt':
                        subtitles.append(self.extract_text(fpath))
                        subtitle_dirs.add(root)
                    elif file.lower().endswith('.pdf') and "reading" in root.lower():
                        readings.append(self.gemini_summarise(self.extract_pdf_text(fpath)))
                    elif file.lower().endswith('.pdf'):
                        quizzes.append(self.extract_pdf_text(fpath))
                    elif file.lower().endswith('.txt') and ("lab" in root.lower() or "walkthrough" in root.lower()):
                        labs.append(self.extract_text(fpath))

            lo = module["Learning Objectives"]
            lops_score = self.compute_lops(lo, {"transcripts": subtitles, "quizzes": quizzes, "reading": readings})
            aoas_score = self.compute_aoas(lo, [("quiz.pdf", q) for q in quizzes])
            rdi_score = self.compute_rdi({"videos": bool(subtitles), "quizzes": bool(quizzes), "readings": bool(readings), "labs": bool(labs)})
            learner_lops = self.gemini_generate(self.learner_lops_prompt(mod_name, lops_score))
            learner_rdi = self.gemini_generate(self.learner_rdi_prompt(mod_name, rdi_score))
            learner_aoas = self.gemini_generate(self.learner_aoas_prompt(mod_name, aoas_score))

            instructor_lops = self.gemini_generate(self.instructor_lops_prompt(mod_name, lops_score))
            instructor_rdi = self.gemini_generate(self.instructor_rdi_prompt(mod_name, rdi_score))
            instructor_aoas = self.gemini_generate(self.instructor_aoas_prompt(mod_name, aoas_score))

            results["Modules"].append({
                "Module_Name": mod_name,
                "LOPS": {
                    "Score": lops_score,
                    "Learner Assessment": learner_lops,
                    "Instructor Feedback": instructor_lops
                },
                "RDI": {
                    "Score": rdi_score,
                    "Learner Assessment": learner_rdi,
                    "Instructor Feedback": instructor_rdi
                },
                "AOAS": {
                    "Score": aoas_score,
                    "Learner Assessment": learner_aoas,
                    "Instructor Feedback": instructor_aoas
                }
            })

            total_lops += lops_score * len(lo) / 5
            total_aoas += aoas_score * len(lo)
            total_objectives += len(lo)
            rdi_scores.append(rdi_score)

        # results["Overall_Summary"] = {
        #     "LOPS_Score": round((total_lops / total_objectives) * 5, 2) if total_objectives else 0.0,
        #     "AOAS Score": round(total_aoas / total_objectives, 2) if total_objectives else 0.0,
        #     "Entropy_RDI_Score": round(np.mean(rdi_scores), 2) if rdi_scores else 0.0
        # }

        return results

if __name__ == "__main__":
    course_path = r""
    evaluator = CourseEvaluator(course_path)
    result = evaluator.run()

    with open(os.path.join(course_path, "course_content_eval.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    print("✅ Course evaluation (LOPS + AOAS + RDI) complete.")