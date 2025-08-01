import os
import requests
import json
import re
import gc
import logging
import numpy as np
import time
from typing import List
import spacy
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
import textstat
import torch
from sentence_transformers import SentenceTransformer, util
from sentence_splitter import split_text_into_sentences
from nltk.tokenize import sent_tokenize
from symspellpy import SymSpell
import psutil
from joblib import Parallel, delayed
import argparse
import fitz
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv('api_key.env')

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument('--base_path', required=True, help='Base path to the course folder')
# args = parser.parse_args()


# BASE_PATH = args.base_path
BASE_PATH = r""

def log_memory_usage(msg):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"{msg} - Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def ensure_symspell_dictionary(target_path="frequency_dictionary_en_82_765.txt"):
    if not os.path.exists(target_path):
        logger.info(f"Dictionary file '{target_path}' not found. Downloading...")
        url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
        r = requests.get(url)
        if r.status_code == 200:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
            logger.info(f"Dictionary downloaded and saved to '{target_path}'")
        else:
            raise FileNotFoundError(f"Failed to download frequency dictionary from {url}")
    return target_path

def grammar_error_count(text, nlp=None):
    """
    Returns an integer 'error' count using lightweight pattern/parse-based checks.
    - Sentence case errors (doesn't start with capital letter)
    - Sentences that don't end with proper punctuation
    - Repeated words (the the)
    - Overly long sentences (>40 words)
    - Detects passive voice sentences using spaCy
    """
    errors = 0
    # Use spaCy if available, otherwise load it now
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=['ner','entity_linker','textcat','lemmatizer'])
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Error 1: Not starting with capital
        if sent and sent[0].isalpha() and not sent[0].isupper():
            errors += 1
        # Error 2: Not ending with punctuation
        if sent and sent[-1] not in '.!?':
            errors += 1
        # Error 3: Repeated word
        if re.search(r'\b(\w+)\s+\1\b', sent, flags=re.IGNORECASE):
            errors += 1
        # Error 4: Too long sentence
        if len(sent.split()) > 40:
            errors += 1
    # Passive voice using spaCy
    doc = nlp(text)
    for sent in doc.sents:
        if any(tok.dep_ == 'auxpass' for tok in sent):
            errors += 1
    return errors

def spelling_error_count(text, sym_spell):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    unique_words = set(words)
    misspelled = 0
    for word in unique_words:
        suggestions = sym_spell.lookup(word, verbosity=0, max_edit_distance=2)
        if not suggestions:
            misspelled += 1
    return misspelled

def passive_voice_count(text, nlp):
    chunk_size = 5000
    total_passive = 0
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        try:
            doc = nlp(chunk)
            total_passive += sum(1 for token in doc if token.dep_ == "auxpass")
        except Exception as e:
            logger.warning(f"Error in passive voice detection: {str(e)[:100]}")
    return total_passive

def avg_sentence_length(text, nlp):
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0
        return sum(len(sent.split()) for sent in sentences) / len(sentences)
    except Exception as e:
        logger.warning(f"Error calculating sentence length: {str(e)[:100]}")
        return 0

def calculate_lqs(text, nlp, sym_spell):
    try:
        grammar_errors = grammar_error_count(text, nlp)
        spelling_errors = spelling_error_count(text, sym_spell)
        passive_uses = passive_voice_count(text, nlp)
        avg_sent_len = avg_sentence_length(text, nlp)
        grammar_penalty = grammar_errors * 0.5
        spelling_penalty = spelling_errors * 0.7
        passive_penalty = passive_uses * 0.6
        complexity_penalty = max(0, avg_sent_len - 25) * 0.5
        raw_score = 100 - (grammar_penalty + spelling_penalty + passive_penalty + complexity_penalty)
        return round(max(0, min(5, raw_score / 20)), 2)
    except Exception as e:
        logger.error(f"Error in LQS calculation: {str(e)}")
        return 50.0

def preprocess_transcript(text):
    return ' '.join(text.strip().split())

def get_fm_score(text, model, batch_size=32):
    try:
        if len(text) < 50:
            return 100.0
        sentences = split_text_into_sentences(text, language='en')
        if len(sentences) < 2:
            return 100.0
        max_sentences = 200
        if len(sentences) > max_sentences:
            step = len(sentences) // max_sentences
            sentences = [sentences[i] for i in range(0, len(sentences), step)][:max_sentences]
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings, dim=0)
        sims = util.cos_sim(embeddings[:-1], embeddings[1:]).diagonal()
        avg_coherence = sims.mean().item()
        return round(min(5, avg_coherence * 5), 2)
    except Exception as e:
        logger.error(f"Error in fluency metric calculation: {str(e)}")
        return 70.0

def get_readability_score(text):
    try:
        max_length = 10000
        if len(text) > max_length:
            sample_size = max_length // 3
            beginning = text[:sample_size]
            middle_start = len(text)//2 - sample_size//2
            middle = text[middle_start:middle_start+sample_size]
            end = text[-sample_size:]
            text = beginning + " " + middle + " " + end
        flesch = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        normalized_score = min(5, max(0, flesch / 20))
        return f"Readability Score: {round(normalized_score, 2)} (Grade Level: {grade})"
    except Exception as e:
        return f"Readability Score error: {str(e)[:100]}"

def get_semantic_coverage(text, objectives, model):
    try:
        max_length = 10000
        if len(text) > max_length:
            sample_size = max_length // 3
            beginning = text[:sample_size]
            middle_start = len(text)//2 - sample_size//2
            middle = text[middle_start:middle_start+sample_size]
            end = text[-sample_size:]
            text = beginning + " " + middle + " " + end
        text_embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        objectives_embedding = model.encode(objectives, convert_to_tensor=True, show_progress_bar=False)
        sim_score = util.cos_sim(text_embedding, objectives_embedding).item()
        return f"Semantic Coverage Score: {round(min(5, sim_score * 5), 2)}"
    except Exception as e:
        return f"Semantic Coverage Score error: {str(e)[:100]}"

def process_file(file_path, course_objectives):
    try:
        logger.info(f"Processing file: {file_path}")
        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'entity_ruler', 'entity_linker', 'textcat', 'lemmatizer'])
        nlp.add_pipe('sentencizer')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = ensure_symspell_dictionary()
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        content = re.sub(r"\n+", r" ", content)
        if not content or len(content) < 20:
            return {
                "File": file_path,
                "Status": "Skipped - Empty or too short",
                "Linguistic Quality Score": 0,
                "Fluency Metric Score": 0
            }
        lqs = calculate_lqs(content, nlp, sym_spell)
        fluency = get_fm_score(content, model)
        readability = get_readability_score(content)
        semantic = get_semantic_coverage(content, course_objectives, model)
        del nlp, model, sym_spell
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "File": file_path,
            "Linguistic Quality Score": lqs,
            "Fluency Metric Score": fluency,
            "Readability Score": readability,
            "Semantic Score": semantic
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"File": file_path, "Error": str(e)[:200]}

def process_file_1(file_path, course_objectives):
    try:
        logger.info(f"Processing file: {file_path}")
        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'entity_ruler', 'entity_linker', 'textcat', 'lemmatizer'])
        nlp.add_pipe('sentencizer')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = ensure_symspell_dictionary()
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        content = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                content += page.get_text()
        if not content or len(content) < 20:
            return {
                "File": file_path,
                "Status": "Skipped - Empty or too short",
                "Linguistic Quality Score": 0,
                "Fluency Metric Score": 0
            }
        lqs = calculate_lqs(content, nlp, sym_spell)
        fluency = get_fm_score(content, model)
        readability = get_readability_score(content)
        semantic = get_semantic_coverage(content, course_objectives, model)
        del nlp, model, sym_spell
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "File": file_path,
            "Linguistic Quality Score": lqs,
            "Fluency Metric Score": fluency,
            "Readability Score": readability,
            "Semantic Score": semantic
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return {"File": file_path, "Error": str(e)[:200]}

def get_all_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                txt_files.append(full_path.replace("\\", "/"))
    return txt_files

def get_all_pdf_files(folder_path):
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pdf'):
                full_path = os.path.join(root, file)
                pdf_files.append(full_path.replace("\\", "/"))
    return pdf_files

def process_module(module_name, learning_objectives, transcript_files, reading_files, n_jobs=None):
    start_time = time.time()
    results = []
    logger.info(f"Starting to process module: {module_name} with {len(transcript_files)} files")
    results_1 = Parallel(n_jobs=n_jobs or os.cpu_count()-1)(
        delayed(process_file)(file_path, learning_objectives)
        for file_path in transcript_files
    )
    results.extend(results_1)
    if len(reading_files) > 1:
        logger.info(f"Starting to process module: {module_name} with {len(reading_files)} files")
        results_2 = Parallel(n_jobs=n_jobs or os.cpu_count()-1)(
            delayed(process_file_1)(file_path, learning_objectives)
            for file_path in reading_files
        )
        results.extend(results_2)
    duration = time.time() - start_time
    logger.info(f"Finished module {module_name} in {duration:.2f} seconds")
    return {module_name: results}

# def linguistic_quality_prompt(score):
#     return f"""
#     A learner is evaluating the overall language quality of a course.

#     The course's **Linguistic Quality Score** is {score} out of 5.
#     This score reflects how grammatically correct, professionally written, and well-structured the course content is.

#     Based on this score, write a **1-sentence, clear, neutral-quality implication of the metric with the focus on whether the user should buy the course or not** for the learner about whether the language quality enhances or distracts from the learning experience.
#     Do not mention the score explicitly.

#     Output only the final sentence.
#     Output format: "1 Line Assessment"
#     """

# def fluency_prompt(score):
#     return f"""
#     A learner is checking whether the course content flows well and feels natural to read.

#     The **Fluency Score** is {score} out of 5, where higher means better readability and flow across sentences and concepts.

#     Write a **1-sentence, clear, neutral-quality implication of the metric with the focus on whether the user should buy the course or not** explaining whether the fluency will help or hinder comprehension.
#     Keep the tone clear, neutral, and avoid mentioning the score.

#     Output only the recommendation sentence.
#     Output format: "1 Line Assessment"
#     """

# def semantic_coverage_prompt(score):
#     return f"""
#     A learner wants to know how well the course explains its topics and whether the content meaning is clear and complete.

#     The **Semantic Coverage Score** is {score} out of 5, indicating how well the material conveys ideas and maintains conceptual clarity.

#     Provide a **1-sentence, clear, neutral-quality implication of the metric with the focus on whether the user should buy the course or not** that helps the learner decide if the material explains topics well enough for understanding and retention.
#     Avoid technical jargon and do not mention the numeric score.

#     Output only the recommendation.
#     Output format: "1 Line Assessment"
#     """

# def readability_prompt(score):
#     return f"""
#     A learner wants to understand whether the course material is written in a way that's easy to read and absorb.

#     The **Readability Score** is {score} out of 5, based on sentence complexity, vocabulary, and clarity.

#     Write a **1-sentence, clear, neutral-quality implication of the metric with the focus on whether the user should buy the course or not** that helps the learner assess how reader-friendly the material is—particularly for non-expert learners.
#     Keep it straightforward and avoid using or referring to the score.

#     Output only the recommendation sentence.
#     Output format: "1 Line Assessment"
#     """

def generate_learner_feedback_for_metric(module_name, metric_name, metric_value):
    metric_definitions = {
        "Linguistic Quality": "Grammatical correctness, sentence structure, and appropriate language use.",
        "Fluency": "How naturally and smoothly the content reads, including phrasing and flow.",
        "Semantic Coverage": "How thoroughly the content communicates the intended learning ideas and concepts.",
        "Readability": "How easy the content is to follow, based on sentence complexity, clarity, and vocabulary.",
    }

    step1 = f"""
You are reviewing the module titled "{module_name}" from a learner’s perspective.

Metric: **{metric_name}**  
Definition: {metric_definitions[metric_name]}  
Score: {metric_value} (scale of 0–5)

Step 1:  
Write a short 3–4 sentence summary that reflects how a thoughtful learner might evaluate the overall quality of the content based on these characteristics. Focus on:
- Whether the content feels manageable to read
- Whether the ideas are clear and well-explained
- Whether the language flows smoothly or might confuse learners
Avoid referencing any metric names or scores directly.
"""
    analysis = model.generate_content(step1).text.strip()

    step2 = f"""
Convert the analysis below into a learner-style feedback message.

To guide tone and style, think of how a learner migh describe the experience.

Requirements:
- Use learner-style phrasing, but without using "I" or "we"
- Passive voice should be used where possible
- Avoid expert tone or generic filler like "it feels like" or "seems to be"
- Language should sound natural, relaxed, and confident — not overly formal
- Focus on clarity, usefulness, and whether learning can actually happen based on the objective
- Strictly Do NOT include any introductory statements

Analysis:
{analysis}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback


def generate_instructor_feedback_for_metric(module_name, metric_name, metric_value):
    metric_definitions = {
        "Linguistic Quality": "Grammar, sentence structure, punctuation, and appropriate language use.",
        "Fluency": "Smoothness of reading, logical flow, and coherence.",
        "Semantic Coverage": "Reflection of key learning ideas and objectives.",
        "Readability": "Ease of comprehension based on vocabulary, syntax, and clarity.",
    }

    step1 = f"""
You are reviewing the module titled "{module_name}" to help the instructor improve the following dimension:

Metric: **{metric_name}**  
Definition: {metric_definitions[metric_name]}  
Score: {metric_value} (scale of 0–5)

Step 1:  
Write a focused diagnostic describing:
- What’s working well
- What could be improved
- How it might affect learner experience


Avoid referencing the score directly. Be constructive and specific.
"""
    diagnostics = model.generate_content(step1).text.strip()

    step2 = f"""
Write a short, professional message for the instructor based on this diagnostic.

Guidelines:
- Do NOT reference metric names or numbers directly.
- Focus on concrete instructional insights: strengths and needed improvements.
- Tone: Professional, respectful, and improvement-oriented.
- Strictly Do NOT include introductory statements.
- Length: Strictly 3–4 sentences.

Instructor Diagnostic:
{diagnostics}
"""
    feedback = model.generate_content(step2).text.strip()
    return feedback

def generate_module_feedback(module_name, semantic_score, readability_score, fluency_score, linguistic_quality, hybrid_score, persona_type: str, focus_metric: str):
    feedback = {}

    if persona_type == 'learner':
        common_guidelines = """
Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, linguistic accessibility, and how comprehensively topics were covered.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").
"""

        prompts = {
            "Linguistic Quality": f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality.

Focus only on: **Linguistic Quality**, reflecting how clear, grammatically sound, and well-written the module felt during the learning process.

**Underlying Quality Observation (used implicitly):**
- Linguistic Quality Score: {linguistic_quality}/5

{common_guidelines}
""",

            "Fluency": f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality.

Focus only on: **Fluency**, describing the smoothness of content progression and clarity of transitions between ideas and sections.

**Underlying Quality Observation (used implicitly):**
- Fluency Score: {fluency_score}/5

{common_guidelines}
""",

            "Semantic Coverage": f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality.

Focus only on: **Semantic Coverage**, expressing how well the module content matched the promised themes and learning expectations based on earlier descriptions.

**Underlying Quality Observation (used implicitly):**
- Semantic Coverage Score: {semantic_score}/5

{common_guidelines}
""",

            "Readability": f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality.

Focus only on: **Readability**, capturing how easy the module was to read and understand, based on the accessibility of the language.

**Underlying Quality Observation (used implicitly):**
- Readability Score: {readability_score}

{common_guidelines}
""",

            "Final Rating": f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality.

Focus on: **Final Rating (Hybrid Score)** — your integrated impression of how coherent, clear, and engaging the module was.

**Underlying Quality Observation (used implicitly):**
- Final Rating: {hybrid_score}/5

{common_guidelines}
"""
        }

    elif persona_type == 'instructor':
        common_guidelines = """
Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, based solely on the underlying quality signals.
* **Avoid:** Direct references to metric names, numerical values, or overly technical instructional design language.
"""

        prompts = {
            "Linguistic Quality": f"""
You are an expert pedagogical consultant reviewing the module "{module_name}".

Focus only on: **Linguistic Quality**, evaluating clarity, grammar, structure, and tone of written content across instructional components.

**Underlying Quality Observation (used implicitly):**
- Linguistic Quality Score: {linguistic_quality}/5

{common_guidelines}
""",

            "Fluency": f"""
You are an expert pedagogical consultant reviewing the module "{module_name}".

Focus only on: **Fluency**, assessing how clearly and logically ideas progressed throughout the instructional material.

**Underlying Quality Observation (used implicitly):**
- Fluency Score: {fluency_score}/5

{common_guidelines}
""",

            "Semantic Coverage": f"""
You are an expert pedagogical consultant reviewing the module "{module_name}".

Focus only on: **Semantic Coverage**, evaluating how comprehensively the module addresses the intended subject matter and whether it fulfills the stated learning goals.

**Underlying Quality Observation (used implicitly):**
- Semantic Coverage Score: {semantic_score}/5

{common_guidelines}
""",

            "Readability": f"""
You are an expert pedagogical consultant reviewing the module "{module_name}".

Focus only on: **Readability**, analyzing how accessible the text was for learners in terms of comprehension and grade-level complexity.

**Underlying Quality Observation (used implicitly):**
- Readability Score: {readability_score}

{common_guidelines}
""",

            "Final Rating": f"""
You are an expert pedagogical consultant reviewing the module "{module_name}".

Focus on: **Final Rating (Hybrid Score)** — a synthesized view of instructional clarity, linguistic quality, and structural coherence.

**Underlying Quality Observation (used implicitly):**
- Final Rating: {hybrid_score}/5

{common_guidelines}
"""
        }

    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")

    # Generate feedback per metric
    prompt = prompts[focus_metric]
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_course_feedback(course_name, semantic_score, readability_score, fluency_score, linguistic_quality, hybrid_score, persona_type: str, focus_metric: str):
    feedback = {}

    if persona_type == 'learner':
        common_guidelines = """
Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").
"""

        prompts = {
            "Linguistic Quality": f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality.

Focus only on: **Linguistic Quality**, which reflects grammatical clarity, coherence of expression, and professionalism of writing across course materials.

**Underlying Quality Observation (used implicitly):**
- Linguistic Quality Score: {linguistic_quality}/5

{common_guidelines}
""",

            "Fluency": f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality.

Focus only on: **Fluency**, which indicates how smoothly ideas and content flowed across the course, supporting intuitive learning progression.

**Underlying Quality Observation (used implicitly):**
- Fluency Score: {fluency_score}/5

{common_guidelines}
""",

            "Semantic Coverage": f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality.

Focus only on: **Semantic Coverage**, which reflects how thoroughly and meaningfully the course addressed the key ideas, themes, or expectations implied by its learning objectives and syllabus.

**Underlying Quality Observation (used implicitly):**
- Semantic Coverage Score: {semantic_score}/5

{common_guidelines}
""",

            "Readability": f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality.

Focus only on: **Readability**, which represents the accessibility of language used — i.e., how easily the course can be read and understood.

**Underlying Quality Observation (used implicitly):**
- Readability Score: {readability_score}

{common_guidelines}
""",

            "Final Rating": f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality.

Focus on the **Final Rating (Hybrid Score)** — a holistic impression of the course based on structure, clarity, coherence, and engagement.

**Underlying Quality Observation (used implicitly):**
- Final Rating: {hybrid_score}/5

{common_guidelines}
"""
        }

    elif persona_type == 'instructor':
        common_guidelines = """
Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, based solely on the underlying quality signals.
* **Avoid:** Direct references to metric names, numerical values, or overly technical instructional design language.
"""

        prompts = {
            "Linguistic Quality": f"""
You are an expert pedagogical consultant reviewing the course "{course_name}".

Focus only on: **Linguistic Quality**, which reflects clarity, coherence, and correctness of the language used across all course materials.

**Underlying Quality Observation (used implicitly):**
- Linguistic Quality Score: {linguistic_quality}/5

{common_guidelines}
""",

            "Fluency": f"""
You are an expert pedagogical consultant reviewing the course "{course_name}".

Focus only on: **Fluency**, which captures the logical flow and smooth progression of content and ideas across modules.

**Underlying Quality Observation (used implicitly):**
- Fluency Score: {fluency_score}/5

{common_guidelines}
""",

            "Semantic Coverage": f"""
You are an expert pedagogical consultant reviewing the course "{course_name}".

Focus only on: **Semantic Coverage**, which reflects the extent to which the course thoroughly develops and delivers the core concepts, skills, and expectations set out in the learning objectives and syllabus.

**Underlying Quality Observation (used implicitly):**
- Semantic Coverage Score: {semantic_score}/5

{common_guidelines}
""",

            "Readability": f"""
You are an expert pedagogical consultant reviewing the course "{course_name}".

Focus only on: **Readability**, indicating the accessibility and complexity of text based on grade-level readability.

**Underlying Quality Observation (used implicitly):**
- Readability Score: {readability_score}

{common_guidelines}
""",

            "Final Rating": f"""
You are an expert pedagogical consultant reviewing the course "{course_name}".

Focus on: **Final Rating (Hybrid Score)** — a comprehensive view of instructional effectiveness, content coherence, and learner engagement.

**Underlying Quality Observation (used implicitly):**
- Final Rating: {hybrid_score}/5

{common_guidelines}
"""
        }

    else:
        raise ValueError("Invalid persona_type. Choose 'learner' or 'instructor'.")

    # Generate feedback for each metric
    prompt = prompts[focus_metric]
    response = model.generate_content(prompt)
    return response.text.strip()


def execute_prompt(prompt):
    response = "None"
    for _ in range(3):
        try:
            output = model.generate_content(prompt)
            response = output.text.strip()
            break
        except InternalServerError as e:
            print("Internal Server Error, retrying...")
            time.sleep(3)
        except ResourceExhausted as r:
            print("Resource Exhausted Error, retrying...")
            time.sleep(3)
        except Exception as e:
            time.sleep(3)
    return response

def get_course_quality_check():
    BASE_COURSE_PATH = BASE_PATH
    os.chdir(BASE_COURSE_PATH)
    log_memory_usage("Starting course quality check")
    try:
        with open('metadata.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return {"error": f"Failed to load metadata: {str(e)}"}

    modules_to_process = []
    course_name = data.get("Course", "Unknown Course")
    for key, value in data.items():
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            logger.info(f"Preparing module: {module_name}")
            learning_objectives = value.get("Learning Objectives", [])
            learning_objectives = "|".join(learning_objectives)
            week_match = re.search(r"(Week|Module)\s+\d+", module_name, re.IGNORECASE)
            if not week_match:
                logger.warning(f"Could not find valid week folder for module '{module_name}'. Skipping.")
                continue
            week_folder = week_match.group(0)
            transcript_files = get_all_txt_files(week_folder)
            reading_files = get_all_pdf_files(week_folder)
            logger.info(f"Found {len(transcript_files)} transcript files in {module_name}")
            logger.info(f"Found {len(reading_files)} reading files in {module_name}")
            modules_to_process.append((module_name, learning_objectives, transcript_files, reading_files))

    all_results = {}
    course_scores = {
        "Linguistic Quality": [],
        "Fluency": [],
        "Semantic Coverage": [],
        "Readability": []
    }

    for module_name, learning_objectives, transcript_files, reading_files in modules_to_process:
        module_result = process_module(module_name, learning_objectives, transcript_files, reading_files, n_jobs=os.cpu_count()-1)
        file_results = module_result[module_name]

        for file_result in file_results:
            if "Linguistic Quality Score" in file_result and isinstance(file_result["Linguistic Quality Score"], (int, float)):
                file_path = file_result["File"]
                folder_name = os.path.basename(os.path.dirname(file_path))
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                if file_path.endswith(".txt"):
                    file_result["File"] = f"Transcript: {folder_name}/{base_name}"
                elif file_path.endswith(".pdf"):
                    file_result["File"] = f"Reading: {base_name}"

                lqs = round(float(file_result["Linguistic Quality Score"]), 2)
                fm = round(float(file_result["Fluency Metric Score"]), 2)

                semantic_match = re.search(r"(\d+(\.\d+)?)", file_result.get("Semantic Score", ""))
                semantic = round(float(semantic_match.group(1)), 2) if semantic_match else 0.0

                readability_match = re.search(r"Readability Score:\s*(\d+(\.\d+)?)", file_result.get("Readability Score", ""))
                readability = round(float(readability_match.group(1)), 2) if readability_match else 0.0

                file_result["Assessment Summary"] = {}

                for metric_name, score in {
                    "Linguistic Quality": lqs,
                    "Fluency": fm,
                    "Semantic Coverage": semantic,
                    "Readability": readability,
                }.items():
                    learner_insights = generate_learner_feedback_for_metric(base_name, metric_name, score)
                    instructor_diagnostics = generate_instructor_feedback_for_metric(base_name, metric_name, score)

                    file_result["Assessment Summary"][f"{metric_name} Score"] = {
                        "Value": score,
                        "Detailed Insights": {
                            "Learner Analysis": learner_insights,
                            "Instructor Diagnostics": instructor_diagnostics
                        }
                    }

                course_scores["Linguistic Quality"].append(lqs)
                course_scores["Fluency"].append(fm)
                course_scores["Semantic Coverage"].append(semantic)
                course_scores["Readability"].append(readability)

        module_lqs = np.mean(course_scores["Linguistic Quality"][-len(file_results):])
        module_fm = np.mean(course_scores["Fluency"][-len(file_results):])
        module_semantic = np.mean(course_scores["Semantic Coverage"][-len(file_results):])
        module_readability = np.mean(course_scores["Readability"][-len(file_results):])
        hybrid_score = round(np.mean([module_lqs, module_fm, module_semantic, module_readability]), 2)

        module_result["Module Summary Feedback"] = {
            "Module Level Evaluation": {
                "Assessment Summary": {
                    "Linguistic Quality Score": {
                        "Value": module_lqs,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='learner', focus_metric="Linguistic Quality"),
                            "Instructor Feedback": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='instructor', focus_metric="Linguistic Quality")
                        }
                    },
                    "Fluency Score": {
                        "Value": module_fm,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='learner', focus_metric="Fluency"),
                            "Instructor Feedback": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='instructor', focus_metric="Fluency")
                        }
                    },
                    "Semantic Coverage Score": {
                        "Value": module_semantic,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='learner', focus_metric="Semantic Coverage"),
                            "Instructor Feedback": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='instructor', focus_metric="Semantic Coverage")
                        }
                    },
                    "Readability Score": {
                        "Value": module_readability,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='learner', focus_metric="Readability"),
                            "Instructor Feedback": generate_module_feedback(module_name, module_semantic, module_readability, module_fm, module_lqs, hybrid_score, persona_type='instructor', focus_metric="Readability")
                        }
                    }
                }
            }
        }

        all_results[module_name] = module_result

    avg_lqs = round(np.mean(course_scores["Linguistic Quality"]), 2)
    avg_fm = round(np.mean(course_scores["Fluency"]), 2)
    avg_semantic = round(np.mean(course_scores["Semantic Coverage"]), 2)
    avg_readability = round(np.mean(course_scores["Readability"]), 2)
    course_hybrid_score = round(np.mean([avg_lqs, avg_fm, avg_semantic, avg_readability]), 2)

    all_results["Course Summary Feedback"] ={
            "Course Level Evaluation": {
                "Assessment Summary": {
                    "Linguistic Quality Score": {
                        "Value": avg_lqs,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='learner', focus_metric="Linguistic Quality"),
                            "Instructor Feedback": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='instructor', focus_metric="Linguistic Quality")
                        }
                    },
                    "Fluency Score": {
                        "Value": avg_fm,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='learner', focus_metric="Fluency"),
                            "Instructor Feedback": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='instructor', focus_metric="Fluency")
                        }
                    },
                    "Semantic Coverage Score": {
                        "Value": avg_semantic,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='learner', focus_metric="Semantic Coverage"),
                            "Instructor Feedback": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='instructor', focus_metric="Semantic Coverage")
                        }
                    },
                    "Readability Score": {
                        "Value": avg_readability,
                        "Detailed Insights": {
                            "Learner Perspective Assessment": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='learner', focus_metric="Readability"),
                            "Instructor Feedback": generate_course_feedback(course_name,avg_semantic, avg_readability, avg_fm, avg_lqs, course_hybrid_score, persona_type='instructor', focus_metric="Readability")
                        }
                    }
                }
            }
        }

    log_memory_usage("All modules processed")
    return all_results


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                for device in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(0.8, device)
            except:
                pass
        start_time = time.time()
        result = get_course_quality_check()
        output_file = "Final final  course_quality_results_updated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        duration = time.time() - start_time
        logger.info(f"Results saved to {output_file}. Total processing time: {duration:.2f} seconds")
        total_modules = len(result)
        total_files = sum(len(module_results) for module_results in result.values())
        logger.info(f"Processed {total_files} files across {total_modules} modules")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()