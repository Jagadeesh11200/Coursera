import json
import time
import numpy as np
import re
import ast
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any


# --------- Google Drive Integration Functions ---------

def get_gdrive_service(service_account_file):
    """Create and return a Google Drive service using service account credentials."""
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def get_json_content(service, file_id):
    """Get and parse JSON file content."""
    request = service.files().get_media(fileId=file_id)
    file_content = request.execute()
    if isinstance(file_content, bytes):
        file_content = file_content.decode('utf-8')
    return json.loads(file_content)

def fetch_metadata_json(service, folder_id):
    """Find and fetch the metadata.json file in the specified folder."""
    query = f"'{folder_id}' in parents and name='metadata.json' and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])
    if not files:
        print("metadata.json not found in the folder")
        return None

    metadata_file_id = files[0]['id']
    print(f"Found metadata.json with ID: {metadata_file_id}")

    try:
        metadata_content = get_json_content(service, metadata_file_id)
        return metadata_content
    except Exception as e:
        print(f"Error fetching metadata.json content: {str(e)}")
        return None


# --------- CourseStructureAnalyzer Class ---------

class CourseStructureAnalyzer:
    def __init__(self, gemini_api_key=None):
        """
        Initialize the course structure analyzer with ML models
        """
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        else:
            self.gemini_model = None

    def call_gemini(self, prompt):
        """
        Sends a prompt to the Gemini model and returns the text response.
        """
        if not self.gemini_model:
            raise RuntimeError("Gemini model is not initialized.")
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

    def get_embeddings(self, texts):
        """Generate embeddings for text using transformer model"""
        if isinstance(texts, str):
            texts = [texts]
        return self.embedding_model.encode(texts)

    def calculate_semantic_similarity(self, texts1, texts2):
        """Calculate semantic similarity between two sets of texts"""
        if not texts1 or not texts2:
            return 0.0

        embeddings1 = self.get_embeddings(texts1)
        embeddings2 = self.get_embeddings(texts2)

        similarity_matrix = cosine_similarity(embeddings1, embeddings2)

        max_similarities = similarity_matrix.max(axis=1)
        return float(np.mean(max_similarities)) # Cast to float

    def gemini_semantic_analysis(self, text1, text2, analysis_type):
        """Use Gemini for advanced semantic analysis"""
        if not self.gemini_model:
            return 0.5  # Default score if Gemini not available

        prompt = f"""
        Analyze the semantic relationship between these two educational content pieces:

        Content 1: {text1}
        Content 2: {text2}

        Analysis Type: {analysis_type}

        Rate the semantic alignment on a scale of 0.0 to 1.0, where:
        - 0.0 = No semantic relationship
        - 0.5 = Moderate relationship
        - 1.0 = Strong semantic alignment

        Consider educational progression, conceptual coherence, and learning flow.
        Return only the numerical score.
        """
        try:
            response = self.gemini_model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        except Exception:
            return 0.5  # Fallback score

    def metric1_inter_module_progression(self, modules):
        """
        Metric 1: Inter-Module Progression Coherence
        Measures smooth transition between consecutive modules
        """
        if len(modules) < 2:
            return [], [] # Return empty lists for scores and intermediate data

        progression_scores = []
        intermediate_data = []

        for i in range(len(modules) - 1):
            current_los = modules[i].get('learning_objectives', [])
            next_los = modules[i + 1].get('learning_objectives', [])

            # Combine current and next module names for context in prompt
            current_module_name = modules[i].get('module_name', f"Module {i+1}")
            next_module_name = modules[i+1].get('module_name', f"Module {i+2}")

            # Calculate embedding similarity
            embedding_score = self.calculate_semantic_similarity(current_los, next_los)

            # Use Gemini for advanced analysis
            gemini_score = self.gemini_semantic_analysis(
                ' '.join(current_los),
                ' '.join(next_los),
                "inter-module progression coherence"
            )

            # Combine scores (weighted average)
            combined_score = float(0.7 * embedding_score + 0.3 * gemini_score) # Cast to float
            progression_scores.append(combined_score)

            # Capture intermediate data
            intermediate_data.append({
                "module_pair": f"{current_module_name} to {next_module_name}",
                "embedding_score": float(embedding_score),
                "gemini_semantic_score": float(gemini_score)
            })

        return progression_scores, intermediate_data

    def dynamic_stopwords_prompt(self):
        return """
You are an expert language model assisting in the development of a course analysis tool.
This tool needs to identify important domain-specific concepts from educational texts such as
learning objectives, course descriptions, and module outlines.

Your task is to generate a comprehensive set of **stop words** — i.e., words that are generally uninformative
for extracting meaningful or content-rich keywords in an academic or educational setting.

The process should be as follows:
1. Internally (without displaying it), generate **guidelines** for what kinds of words should be considered stop words.
Use your knowledge of linguistic structure and instructional content to guide this.
- Include common function words (e.g., 'is', 'the', 'was', 'on')
- Include generic instructional verbs (e.g., 'understand', 'learn', 'know', 'explore', 'discuss')
- Include vague academic terms (e.g., 'topic', 'concept', 'idea', 'content')
- Include non-specific quantifiers (e.g., 'many', 'some', 'various', 'several')
- Include words that are often used across different domains but rarely convey key topical meaning
- Include any other types you deem appropriate based on your internal reasoning

2. Based strictly on the internally created guidelines, generate a **comprehensive, domain-agnostic list**
of such stop words. The list should be reasonably exhaustive but should not include specific subject terms
like 'photosynthesis', 'calculus', etc. Avoid over-inclusion of valid content words.

Output Requirements:
- Return only the final result: a valid Python `set` of lowercase words.
- Format the output as a Python code literal (e.g., {'this', 'that', 'with'})
- Do not include any explanation, markdown formatting, comments, or additional metadata.
- The output should be directly parsable using ast.literal_eval() in Python.
"""

    def extract_key_concepts(self, text):
        """Extract key concepts from text for coverage analysis"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Static fallback stop word set
        fallback_stop_words = {
            'this', 'that', 'with', 'from', 'they', 'been', 'have', 'were', 'said', 'each',
            'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other',
            'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also',
            'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being',
            'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some',
            'these', 'many', 'then', 'them', 'well'
        }

        # Try LLM-based stop word generation
        try:
            llm_output = self.call_gemini(self.dynamic_stopwords_prompt())
            stop_words = ast.literal_eval(llm_output)
            if not isinstance(stop_words, set):
                raise ValueError("Output is not a set")
        except Exception:
            stop_words = fallback_stop_words

        # Filter keywords
        return [word for word in set(words) if word not in stop_words]

    def metric2_course_module_alignment(self, course_info, modules):
        """
        Metric 2: Course-Module Alignment Score (FINAL)
        Evaluates whether combined learning objectives across ALL modules
        collectively fulfill the course's overall promise — returns a single score.
        Uses dynamic criteria generation and a fixed scoring rubric, with chain-of-prompts.
        """
        course_description = course_info.get('About this Course', '')
        course_title = course_info.get('Course', '')
        course_promise = f"{course_title}. {course_description}".strip()

        all_learning_objectives = []
        for module in modules:
            all_learning_objectives.extend(module.get('learning_objectives', []))

        if not all_learning_objectives:
            return 1.0, {"steps": ["No learning objectives found, defaulting to perfect alignment."]}

        intermediate_data = {"steps": []}

        # Step 1: Dynamic criteria generation
        dynamic_criteria_prompt = f"""
        You are a course evaluation expert.

        Given the following course promise:
        \"\"\"{course_promise}\"\"\"

        Generate a concise, expert-level list of specific, measurable, achievable, relevant, and time-bound (SMART) evaluation criteria that define what this course must deliver to be considered a success.
        Each criterion should clearly reflect an essential aspect of what the course advertises, such as:
        - Coverage of promised topics or skills
        - Appropriate level of depth or complexity
        - Development of practical or transferable skills
        - Coherent alignment with student learning outcomes
        - Specific knowledge or ability acquisition.

        Avoid generic or redundant statements. Ensure each criterion is distinct and actionable, and captures a unique expectation conveyed by the course promise.

        Format your output as a numbered list of clear and concise criteria.
        """
        try:
            criteria_response = self.gemini_model.generate_content(dynamic_criteria_prompt)
            generated_criteria = criteria_response.text.strip()

        except Exception as e:
            generated_criteria = (
                "1. Covers all promised topics.\n"
                "2. Provides sufficient depth of learning.\n"
                "3. Develops applicable skills.\n"
                "4. Aligns learning objectives to course goals.\n"
                "5. Ensures knowledge acquisition at the stated level."
            )
            intermediate_data["steps"].append({
                "step_name": "Dynamic Criteria Generation",
                "status": "Failed",
                "details": f"Gemini failed to generate criteria, using fallback. Error: {e}"
            })

        # Step 2: Detailed evaluation prompt against criteria
        evaluation_prompt = f"""
        You are evaluating whether a course’s learning objectives align with its promised outcomes.

        COURSE PROMISE:
        \"\"\"{course_promise}\"\"\"

        COMBINED LEARNING OBJECTIVES:
        {chr(10).join(f"- {obj}" for obj in all_learning_objectives)}

        EVALUATION CRITERIA (derived from the course promise):
        {generated_criteria}

        Based on the combined learning objectives, assess how well the course satisfies *each* of the EVALUATION CRITERIA.
        For each criterion, provide a brief justification (1-2 sentences) indicating why it is met, partially met, or not met by the learning objectives.

        Finally, based on your assessment of *all* criteria, assign a single overall numerical score using the fixed rubric below.
        Consider:
        - **Coverage completeness**: Do the objectives clearly cover the major themes, skills, and goals stated in the course promise?
        - **Depth alignment**: Is the level of detail or complexity in the objectives consistent with what the course promises to teach?
        - **Skill development**: Will learners gain the practical, technical, or cognitive skills implied by the promise?
        - **Coherence and integration**: Do the objectives collectively form a clear, focused, and well-aligned learning path that supports the course's intent?

        Fixed Rubric:
        - **1.0** = Objectives comprehensively fulfill all course promises and criteria.
        - **0.7** = Objectives strongly align with most course promises and criteria, with minor gaps.
        - **0.5** = Objectives moderately align with course promises and criteria, addressing key elements but with notable omissions.
        - **0.3** = Objectives partially address some promises and criteria but miss many key elements.
        - **0.0** = Objectives completely fail to deliver on course promises and criteria.

        Format your output as follows:
        <CRITERIA_ASSESSMENT>
        [Justification for each criterion, e.g., "Criterion 1: Met. Justification..."]
        </CRITERIA_ASSESSMENT>
        <OVERALL_SCORE>
        [Numerical score only, e.g., 0.7]
        </OVERALL_SCORE>
        """

        gemini_score = 0.5
        criteria_assessment_text = ""
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(evaluation_prompt)
                response_text = response.text.strip()

                # Extract score and assessment using regex
                score_match = re.search(r"<OVERALL_SCORE>\s*(\d+\.?\d*)\s*</OVERALL_SCORE>", response_text, re.DOTALL)
                assessment_match = re.search(r"<CRITERIA_ASSESSMENT>(.*?)</CRITERIA_ASSESSMENT>", response_text, re.DOTALL)

                if score_match:
                    gemini_score = float(score_match.group(1))
                    gemini_score = max(0.0, min(1.0, gemini_score))
                if assessment_match:
                    criteria_assessment_text = assessment_match.group(1).strip()


            except Exception as e:
                intermediate_data["steps"].append({
                    "step_name": "Gemini Evaluation",
                    "status": "Failed",
                    "details": f"Gemini evaluation failed, using fallback score. Error: {e}"
                })
                gemini_score = 0.5 # Fallback score

        # Step 3: Semantic similarity and keyword coverage (existing logic)
        embedding_score = self.calculate_semantic_similarity(
            [course_promise], all_learning_objectives
        )
        intermediate_data["steps"].append({
            "step_name": "Embedding Similarity Calculation",
            "details": f"Embedding similarity between course promise and objectives: {embedding_score:.2f}"
        })

        course_keywords = self.extract_key_concepts(course_promise)
        objective_keywords = self.extract_key_concepts(' '.join(all_learning_objectives))
        coverage_score = len(set(course_keywords) & set(objective_keywords)) / max(len(course_keywords), 1)
        intermediate_data["steps"].append({
            "step_name": "Keyword Coverage Calculation",
            "details": {
                "coverage_score": f"{coverage_score:.2f}"
            }
        })

        # Final score: weighted average
        final_score = float(0.4 * embedding_score + 0.4 * gemini_score + 0.2 * coverage_score) # Cast to float
        intermediate_data["final_score_components"] = {
            "embedding_component": f"{0.4 * embedding_score:.2f}",
            "gemini_component": f"{0.4 * gemini_score:.2f}",
            "keyword_coverage_component": f"{0.2 * coverage_score:.2f}"
        }
        intermediate_data["gemini_criteria_assessment"] = criteria_assessment_text

        return final_score, intermediate_data


    def metric3_intra_module_content_flow(self, modules):
        """
        Metric 3: Intra-Module Content Flow
        Assesses logical progression within module syllabus
        """
        flow_scores = []
        intermediate_data = []

        for module in modules:
            syllabus = module.get('syllabus', [])
            module_name = module.get('module_name', 'Unnamed Module')

            if len(syllabus) < 2:
                flow_scores.append(1.0)
                intermediate_data.append({
                    "module_name": module_name,
                    "syllabus_items": syllabus,
                    "detail": "Single syllabus item, perfect flow assumed.",
                    "sequential_similarities": []
                })
                continue

            sequential_similarities = []
            for i in range(len(syllabus) - 1):
                similarity = self.calculate_semantic_similarity(
                    [syllabus[i]], [syllabus[i + 1]]
                )
                sequential_similarities.append(float(similarity)) # Cast to float

            avg_flow = float(np.mean(sequential_similarities)) # Cast to float
            flow_scores.append(avg_flow)
            intermediate_data.append({
                "module_name": module_name,
                "syllabus_items": syllabus,
                "sequential_similarities": sequential_similarities # Already cast above
            })
        return flow_scores, intermediate_data

    def metric4_objective_content_alignment(self, modules):
        """
        Metric 4: Objective-Content Alignment
        Measures how well syllabus supports learning objectives
        """
        alignment_scores = []
        intermediate_data = []

        for module in modules:
            los = module.get('learning_objectives', [])
            syllabus = module.get('syllabus', [])
            module_name = module.get('module_name', 'Unnamed Module')

            if not los or not syllabus:
                alignment_scores.append(0.0) # No objectives or syllabus means no alignment to measure
                intermediate_data.append({
                    "module_name": module_name,
                    "detail": "Missing learning objectives or syllabus, alignment cannot be assessed.",
                    "embedding_score": 0.0,
                    "gemini_semantic_score": 0.0
                })
                continue

            embedding_score = self.calculate_semantic_similarity(los, syllabus)

            gemini_score = self.gemini_semantic_analysis(
                ' '.join(los),
                ' '.join(syllabus),
                "objective-content alignment"
            )

            combined_score = float(0.65 * embedding_score + 0.35 * gemini_score) # Cast to float
            alignment_scores.append(combined_score)
            intermediate_data.append({
                "module_name": module_name,
                "embedding_score": float(embedding_score),
                "gemini_semantic_score": float(gemini_score)
            })

        return alignment_scores, intermediate_data

    def metric5_module_learning_unity(self, modules):
        """
        Metric 5: Module Learning Unity
        Evaluates internal consistency within modules
        """
        unity_scores = []
        intermediate_data = []

        for module in modules:
            los = module.get('learning_objectives', [])
            syllabus = module.get('syllabus', [])
            module_name = module.get('module_name', 'Unnamed Module')

            # Multi-faceted unity analysis
            # Coherence of LOs among themselves
            lo_coherence = self.calculate_semantic_similarity(los, los) if len(los) > 1 else 1.0
            # Coherence of syllabus items among themselves
            content_coherence = self.calculate_semantic_similarity(syllabus, syllabus) if len(syllabus) > 1 else 1.0
            # Alignment of module name with its LOs and syllabus
            name_alignment = self.calculate_semantic_similarity([module_name], los + syllabus) if (los or syllabus) else 0.0 # Handle empty LOs/Syllabus


            # Weighted unity score
            unity_score = float(0.4 * lo_coherence + 0.4 * content_coherence + 0.2 * name_alignment) # Cast to float
            unity_scores.append(unity_score)
            intermediate_data.append({
                "module_name": module_name,
                "lo_coherence_score": float(lo_coherence),
                "content_coherence_score": float(content_coherence),
                "name_alignment_score": float(name_alignment)
            })
        return unity_scores, intermediate_data

    # Define the prompt functions as methods within the class
    def prompt_inter_module_progression_coherence(self, intermediate_gemini_semantic_score, intermediate_embedding_score, final_rating, module_pair=None):
        """
        Generates tailored prompts for the "Inter-Module Progression Coherence" metric.
        Handles both module-level and course-level prompts.
        """
        if module_pair:
            # Module-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

This reflection focuses on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience. Your output should directly convey feelings and observations about the module's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Inter-Module Progression Coherence (Score Range: 1-5):**
    * **Definition:** This metric assesses how smoothly one module transitions into the next, ensuring a logical and understandable progression of learning.
    * **Scoring:** 1 (disjointed or confusing transitions) to 5 (seamless and intuitive progression).

* **Semantic Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This indicates the logical flow and conceptual coherence between modules, going beyond just keyword matching.
    * **Scoring:** 0.0 (no semantic relationship) to 1.0 (strong semantic alignment).

* **Conceptual Closeness (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures how conceptually close the learning objectives of consecutive modules are, indicating ease of connecting ideas.
    * **Scoring:** 0.0 (distant concepts) to 1.0 (very close concepts).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Inter-Module Progression Coherence: {final_rating}/5
- Semantic Coherence: {intermediate_gemini_semantic_score:.3f}/1.0
- Conceptual Closeness: {intermediate_embedding_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me) for the overall description.** For specific feelings/observations, use "This felt..." or "The experience was...".
* **Style:** 2–4 sentences, describing the personal experience.
* **Focus:** Perceived clarity, logical flow, and ease of connecting ideas between modules, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Inter-Module Progression Coherence: {final_rating}, Semantic Coherence: {intermediate_gemini_semantic_score:.3f}, and Conceptual Closeness: {intermediate_embedding_score:.3f}, write a short learner-style reflection on the transition from **'{module_pair.split(' to ')[0]}'** to **'{module_pair.split(' to ')[1]}'**. Simulate how a learner experienced engaging with the content and structure between these two modules. The last sentence should be a concise summary of the overall impression of the transition's effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""

            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the module's syllabus by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall course structure. My aim is to pinpoint specific design strengths within the syllabus and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Inter-Module Progression Coherence (Score Range: 1-5):**
    * **Definition:** This metric assesses the design quality of transitions between consecutive modules, focusing on the logical and conceptual scaffolding provided to learners.
    * **Scoring:** 1 (disjointed or confusing transitions requiring significant learner effort) to 5 (seamless, intuitively structured progression that minimizes cognitive load).

* **Semantic Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation provides deep insights into the pedagogical flow and conceptual scaffolding between modules, precisely identifying potential gaps, redundancies, or abrupt shifts.
    * **Scoring:** 0.0 (no discernible pedagogical relationship) to 1.0 (superior, deliberately structured progression).

* **Conceptual Closeness (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This metric quantifies the semantic overlap between consecutive modules' learning objectives, directly indicating the conceptual distance.
    * **Scoring:** 0.0 (minimal conceptual overlap) to 1.0 (high conceptual overlap, critical for minimizing learner cognitive load and enhancing engagement).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Inter-Module Progression Coherence: {final_rating}/5
- Semantic Coherence: {intermediate_gemini_semantic_score:.3f}/1.0
- Conceptual Closeness: {intermediate_embedding_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Inter-Module Progression Coherence: {final_rating}, Semantic Coherence: {intermediate_gemini_semantic_score:.3f}, and Conceptual Closeness: {intermediate_embedding_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the transition from **'{module_pair.split(' to ')[0]}'** to **'{module_pair.split(' to ')[1]}'**. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""
        else:
            # Course-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's interconnected elements (overall goals, content across modules, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience. The output should directly convey feelings and observations about the course's effectiveness as a whole.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**

* **Course Progression Coherence (Score Range: 1-5):**
    * **Definition:** This metric provides a holistic assessment of how smoothly the entire course progresses from one module to the next, ensuring a logical and understandable flow of learning across the curriculum.
    * **Scoring:** 1 (disjointed or confusing progression that hinders learning) to 5 (a seamless and intuitive learning journey from start to finish).

* **Semantic Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This indicates the logical flow and conceptual coherence between all modules, going beyond just keyword matching to assess the overall pedagogical progression of the course.
    * **Scoring:** 0.0 (no semantic relationship) to 1.0 (strong semantic alignment).

* **Conceptual Closeness (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures how conceptually close the learning objectives are from one module to the next, averaged across the entire course.
    * **Scoring:** 0.0 (distant concepts) to 1.0 (very close concepts).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Course Progression Coherence: {final_rating}/5
- Semantic Coherence (Average): {intermediate_gemini_semantic_score:.3f}/1.0
- Conceptual Closeness (Average): {intermediate_embedding_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner, reflecting on the *entire course*.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience of the *course's* progression.
* **Focus:** Perceived clarity, logical flow, and ease of connecting ideas across all modules, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Course Progression Coherence: {final_rating}, Semantic Coherence (Average): {intermediate_gemini_semantic_score:.3f}, and Conceptual Closeness (Average): {intermediate_embedding_score:.3f}, write a short learner-style reflection on the overall progression of the course. Simulate how a learner experienced the flow of content and structure from the beginning to the end of the course. The last sentence should be a concise summary of the overall impression of the course's progression effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on the *overall course's* design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact and seamless integration of its components.

This assessment serves to validate the *entire course's* structure by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution of all modules to the overall course structure. The aim is to pinpoint specific design strengths within the overall course and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Course Progression Coherence (Score Range: 1-5):**
    * **Definition:** This metric provides a holistic assessment of the design quality of transitions between *all* consecutive modules, focusing on the logical and conceptual scaffolding provided to learners throughout the course.
    * **Scoring:** 1 (disjointed or confusing progression requiring significant learner effort) to 5 (seamless, intuitively structured progression that minimizes cognitive load across the entire curriculum).

* **Semantic Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation provides deep insights into the pedagogical flow and conceptual scaffolding between all modules, precisely identifying potential gaps, redundancies, or abrupt shifts at a course level.
    * **Scoring:** 0.0 (no discernible pedagogical relationship) to 1.0 (superior, deliberately structured progression).

* **Conceptual Closeness (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This metric quantifies the average semantic overlap between consecutive modules' learning objectives across the entire course, directly indicating the overall conceptual distance.
    * **Scoring:** 0.0 (minimal conceptual overlap) to 1.0 (high conceptual overlap, critical for minimizing learner cognitive load and enhancing engagement).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Course Progression Coherence: {final_rating}/5
- Semantic Coherence (Average): {intermediate_gemini_semantic_score:.3f}/1.0
- Conceptual Closeness (Average): {intermediate_embedding_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor about the *overall course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations for the *entire course*. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the *course's* pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Course Progression Coherence: {final_rating}, Semantic Coherence (Average): {intermediate_gemini_semantic_score:.3f}, and Conceptual Closeness (Average): {intermediate_embedding_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the overall Course Progression Coherence. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths of the *entire course*. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        return user_prompt, instructor_prompt

    def prompt_course_module_alignment(self, intermediate_embedding_component_raw, intermediate_gemini_component_raw, intermediate_keyword_coverage_component_raw, final_rating):
        """
        Generates tailored prompts for the "Course-Module Alignment" metric.
        """
        user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's interconnected elements (overall goals, content across modules, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the course's effectiveness as a whole.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**

* **Course-Module Alignment (Score Range: 1-5):**
    * **Definition:** This metric assesses how well all modules collectively fulfill the course's overall promise and learning goals. This is an evaluation of the course as a whole, not individual modules.
    * **Scoring:** 1 (modules significantly deviate from course promise) to 5 (modules perfectly align with and fulfill course promise).

* **Conceptual Match (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates how well the overall course description (what the course promises) aligns with the combined learning objectives of all modules.
    * **Scoring:** 0.0 (no conceptual match) to 1.0 (strong conceptual match).

* **Intent Fulfillment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation assesses whether the learning objectives truly deliver on the course's stated focus and promises, considering both the intent and how the course is designed.
    * **Scoring:** 0.0 (no intent fulfillment) to 1.0 (robust and trustworthy alignment).

* **Topic Coverage (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures how many important keywords or concepts from the course description are explicitly included in the learning objectives.
    * **Scoring:** 0.0 (minimal topic coverage) to 1.0 (thorough topic coverage).

**Underlying Quality Observations (These influence the perceived experience, but should NOT be stated as values or technical names explicitly in the reflection):**
- Course-Module Alignment: {final_rating}/5
- Conceptual Match: {intermediate_embedding_component_raw:.3f}/1.0
- Intent Fulfillment: {intermediate_gemini_component_raw:.3f}/1.0
- Topic Coverage: {intermediate_keyword_coverage_component_raw:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner, reflecting on the *entire course*.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience of the *course's* coherence.
* **Focus:** Perceived clarity, relevance of content to the overall course, and how comprehensively topics were covered across the *entire course*, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Course-Module Alignment: {final_rating}, Conceptual Match: {intermediate_embedding_component_raw:.3f}, Intent Fulfillment: {intermediate_gemini_component_raw:.3f}, and Topic Coverage: {intermediate_keyword_coverage_component_raw:.3f}, write a short learner-style reflection on the overall course's alignment with its modules. Simulate how a learner experienced the coherence and relevance of the course's content. The last sentence should be a concise summary of the overall impression of the course's effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
        instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on the *overall course's* design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact and seamless integration of its components.

This assessment serves to validate the *entire course's* structure by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution of all modules to the overall course structure. The aim is to pinpoint specific design strengths within the overall course and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Course-Module Alignment (Score Range: 1-5):**
    * **Definition:** This metric provides a holistic assessment of how cohesively all modules contribute to and fulfill the overarching course objectives and promise. This is an evaluation of the course as a whole, not individual modules.
    * **Scoring:** 1 (modules significantly deviate from the course's stated goals) to 5 (modules are optimally designed to collectively achieve and exceed course objectives).

* **Conceptual Match (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This component quantifies the semantic congruence between the overarching course promise (as stated in the description) and the aggregate learning objectives of all modules.
    * **Scoring:** 0.0 (minimal conceptual alignment) to 1.0 (strong, inherent conceptual alignment).

* **Intent Fulfillment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This sophisticated evaluation assesses whether the learning objectives comprehensively and meaningfully address the course's stated goals and intended outcomes, moving beyond superficial keyword matching.
    * **Scoring:** 0.0 (lack of intentional alignment) to 1.0 (robust, intentional alignment).

* **Topic Coverage (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This metric measures the explicit inclusion of important keywords or concepts from the course description within the learning objectives, indicating transparency and direct relevance.
    * **Scoring:** 0.0 (minimal explicit topic coverage) to 1.0 (thorough and direct topic coverage).

**Underlying Quality Observations (These are the precise analytical scores informing expert recommendations; DO NOT state these values or their technical names explicitly in the feedback):**
- Course-Module Alignment: {final_rating}/5
- Conceptual Match: {intermediate_embedding_component_raw:.3f}/1.0
- Intent Fulfillment: {intermediate_gemini_component_raw:.3f}/1.0
- Topic Coverage: {intermediate_keyword_coverage_component_raw:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor about the *overall course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations for the *entire course*. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the *course's* pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Course-Module Alignment: {final_rating}, Conceptual Match: {intermediate_embedding_component_raw:.3f}, Intent Fulfillment: {intermediate_gemini_component_raw:.3f}, and Topic Coverage: {intermediate_keyword_coverage_component_raw:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the overall Course-Module Alignment. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths of the *entire course*. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        return user_prompt, instructor_prompt

    def prompt_intra_module_content_flow(self, intermediate_sequential_similarities_avg, module_name=None, final_rating=None):
        """
        Generates tailored prompts for the "Intra-Module Content Flow" metric.
        Handles both module-level and course-level prompts.
        """
        if module_name:
            # Module-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

This reflection focuses on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the module's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Intra-Module Content Flow (Score Range: 1-5):**
    * **Definition:** This metric assesses the logical progression and coherence of topics and content within a single module.
    * **Scoring:** 1 (disorganized or disjointed content flow) to 5 (smooth, logical, and easy-to-follow content progression).

* **Sequential Similarities (Intermediate Score Range: 0.0-1.0 for each transition):**
    * **Definition:** This measures the conceptual similarity between consecutive syllabus items, indicating how smoothly one topic leads to the next.
    * **Scoring:** 0.0 (no conceptual similarity) to 1.0 (high conceptual similarity).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Intra-Module Content Flow: {final_rating}/5
- Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience.
* **Focus:** Perceived clarity, logical flow, and ease of understanding the progression of topics within the module, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on flow").

**Prompt Instruction:**
Given the underlying quality observations: Intra-Module Content Flow: {final_rating} and Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}, write a short learner-style reflection on the content flow within the module '{module_name}'. Simulate how a learner experienced engaging with the module's content structure. The last sentence should be a concise summary of the overall impression of the module's content flow effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the module's syllabus by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall course structure. My aim is to pinpoint specific design strengths within the syllabus and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Intra-Module Content Flow (Score Range: 1-5):**
    * **Definition:** This metric rigorously assesses the internal logical coherence and sequential progression of topics and activities within a specific module's syllabus.
    * **Scoring:** 1 (disorganized, confusing, or contradictory content flow) to 5 (exceptionally smooth, intuitive, and pedagogically sound content progression).

* **Sequential Similarities (Intermediate Score Range: 0.0-1.0 for each transition):**
    * **Definition:** This quantifies the conceptual continuity between consecutive syllabus items, serving as a direct indicator of the deliberate scaffolding or abrupt shifts in topic.
    * **Scoring:** 0.0 (significant conceptual discontinuity) to 1.0 (seamless conceptual transition).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Intra-Module Content Flow: {final_rating}/5
- Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Intra-Module Content Flow: {final_rating} and Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the content flow within the module '{module_name}'. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""
        else:
            # Course-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the course's effectiveness as a whole.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**

* **Course Content Flow (Score Range: 1-5):**
    * **Definition:** This metric holistically assesses the logical progression and coherence of topics and content *across all modules* of the entire course.
    * **Scoring:** 1 (disorganized or disjointed content flow) to 5 (smooth, logical, and easy-to-follow content progression).

* **Sequential Similarities (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures the average conceptual similarity between consecutive syllabus items across the entire course, indicating how smoothly one topic leads to the next.
    * **Scoring:** 0.0 (no conceptual similarity) to 1.0 (high conceptual similarity).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Course Content Flow: {final_rating}/5
- Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner, reflecting on the *entire course*.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience of the *course's* content flow.
* **Focus:** Perceived clarity, logical flow, and ease of understanding the progression of topics across all modules, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on flow").

**Prompt Instruction:**
Given the underlying quality observations: Course Content Flow: {final_rating} and Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}, write a short learner-style reflection on the overall content flow of the course. Simulate how a learner experienced engaging with the content structure from the beginning to the end. The last sentence should be a concise summary of the overall impression of the course's content flow effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on the *overall course's* design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact and seamless integration of its components.

This assessment serves to validate the *entire course's* structure by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution of all modules to the overall course structure. The aim is to pinpoint specific design strengths within the overall course and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Course Content Flow (Score Range: 1-5):**
    * **Definition:** This metric rigorously assesses the internal logical coherence and sequential progression of topics and activities across *the entire course's curriculum*.
    * **Scoring:** 1 (disorganized, confusing, or contradictory content flow) to 5 (exceptionally smooth, intuitive, and pedagogically sound content progression).

* **Sequential Similarities (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This quantifies the average conceptual continuity between consecutive syllabus items across the entire course, serving as a direct indicator of the deliberate scaffolding or abrupt shifts in topic.
    * **Scoring:** 0.0 (significant conceptual discontinuity) to 1.0 (seamless conceptual transition).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Course Content Flow: {final_rating}/5
- Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor about the *overall course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations for the *entire course*. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the *course's* pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Course Content Flow: {final_rating} and Sequential Similarities (average): {intermediate_sequential_similarities_avg:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the overall content flow of the course. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths of the *entire course*. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        return user_prompt, instructor_prompt

    def prompt_objective_content_alignment(self, intermediate_embedding_score, intermediate_gemini_semantic_score, module_name=None, final_rating=None):
        """
        Generates tailored prompts for the "Objective-Content Alignment" metric.
        Handles both module-level and course-level prompts.
        """
        if module_name:
            # Module-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

This reflection focuses on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the module's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Objective-Content Alignment (Score Range: 1-5):**
    * **Definition:** This metric indicates how precisely the module's stated learning objectives directly corresponded to, and were comprehensively covered by, the actual content provided in the syllabus.
    * **Scoring:** 1 (significant disconnects or content deficiencies) to 5 (perfect congruence and comprehensive delivery).

* **Conceptual Support (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates how well the module's syllabus conceptually supports its stated learning objectives.
    * **Scoring:** 0.0 (minimal support) to 1.0 (strong conceptual support).

* **Semantic Cohesion (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation assesses the deeper semantic coherence between objectives and content, ensuring the content meaningfully addresses the learning goals.
    * **Scoring:** 0.0 (no semantic cohesion) to 1.0 (strong semantic cohesion).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Objective-Content Alignment: {final_rating}/5
- Conceptual Support: {intermediate_embedding_score:.3f}/1.0
- Semantic Cohesion: {intermediate_gemini_semantic_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience.
* **Focus:** Perceived clarity, intellectual engagement, relevance of content to learning objectives, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Objective-Content Alignment: {final_rating}, Conceptual Support: {intermediate_embedding_score:.3f}, and Semantic Cohesion: {intermediate_gemini_semantic_score:.3f}, write a short learner-style reflection on the objective-content alignment within the module '{module_name}'. Simulate how a learner experienced the connection between what was promised to be learned and what was actually provided. The last sentence should be a concise summary of the overall impression of the module's effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the module's syllabus by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall course structure. My aim is to pinpoint specific design strengths within the syllabus and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Objective-Content Alignment (Score Range: 1-5):**
    * **Definition:** This metric quantifies how precisely the module's stated learning objectives directly correspond to, and are comprehensively covered by, the actual content presented in the syllabus.
    * **Scoring:** 1 (significant disconnects or content deficiencies relative to objectives) to 5 (perfect congruence; all objectives are fully supported, clearly addressed, and content is optimally aligned).

* **Conceptual Support (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This component measures the direct semantic support the module's syllabus provides for its stated learning objectives, focusing on the explicit thematic and topical connections.
    * **Scoring:** 0.0 (minimal conceptual support) to 1.0 (strong, evident conceptual support).

* **Semantic Cohesion (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation assesses the deeper, implicit semantic coherence between the learning objectives and the syllabus content, ensuring that the content not only covers but also meaningfully addresses the intended learning outcomes.
    * **Scoring:** 0.0 (lack of meaningful semantic cohesion) to 1.0 (strong, intentional semantic cohesion).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Objective-Content Alignment: {final_rating}/5
- Conceptual Support: {intermediate_embedding_score:.3f}/1.0
- Semantic Cohesion: {intermediate_gemini_semantic_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Objective-Content Alignment: {final_rating}, Conceptual Support: {intermediate_embedding_score:.3f}, and Semantic Cohesion: {intermediate_gemini_semantic_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the objective-content alignment within the module '{module_name}'. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""
        else:
            # Course-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the course's effectiveness as a whole.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Course Objective-Content Alignment (Score Range: 1-5):**
    * **Definition:** This metric provides a holistic assessment of how precisely the course's learning objectives across all modules were directly corresponded to and comprehensively covered by the course's content.
    * **Scoring:** 1 (significant disconnects or content deficiencies) to 5 (perfect congruence and comprehensive delivery across all modules).

* **Conceptual Support (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates how well the course's aggregated syllabus content conceptually supports all its stated learning objectives.
    * **Scoring:** 0.0 (minimal support) to 1.0 (strong conceptual support).

* **Semantic Cohesion (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation assesses the deeper semantic coherence between all course objectives and all syllabus content, ensuring the content meaningfully addresses the learning goals across the entire curriculum.
    * **Scoring:** 0.0 (no semantic cohesion) to 1.0 (strong semantic cohesion).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Course Objective-Content Alignment: {final_rating}/5
- Conceptual Support (Average): {intermediate_embedding_score:.3f}/1.0
- Semantic Cohesion (Average): {intermediate_gemini_semantic_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner, reflecting on the *entire course*.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience of the *course's* alignment.
* **Focus:** Perceived clarity, intellectual engagement, and relevance of content to learning objectives across all modules, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Course Objective-Content Alignment: {final_rating}, Conceptual Support (Average): {intermediate_embedding_score:.3f}, and Semantic Cohesion (Average): {intermediate_gemini_semantic_score:.3f}, write a short learner-style reflection on the overall objective-content alignment of the course. Simulate how a learner experienced the connection between what was promised to be learned and what was actually provided. The last sentence should be a concise summary of the overall impression of the course's effectiveness from a learner's perspective, implicitly offering a key takeaway.
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on the *overall course's* design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact and seamless integration of its components.

This assessment serves to validate the *entire course's* structure by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution of all modules to the overall course structure. The aim is to pinpoint specific design strengths within the overall course and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Course Objective-Content Alignment (Score Range: 1-5):**
    * **Definition:** This metric quantifies how precisely all of the course's stated learning objectives across all modules correspond to and are comprehensively covered by the course's aggregated content.
    * **Scoring:** 1 (significant disconnects or content deficiencies relative to objectives) to 5 (perfect congruence; all objectives are fully supported, clearly addressed, and content is optimally aligned).

* **Conceptual Support (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This component measures the average direct semantic support that all module syllabi provide for their stated learning objectives across the entire course.
    * **Scoring:** 0.0 (minimal conceptual support) to 1.0 (strong, evident conceptual support).

* **Semantic Cohesion (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This advanced evaluation assesses the deeper, implicit semantic coherence between all course objectives and all syllabus content, ensuring that the content not only covers but also meaningfully addresses the intended learning outcomes.
    * **Scoring:** 0.0 (lack of meaningful semantic cohesion) to 1.0 (strong, intentional semantic cohesion).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Course Objective-Content Alignment: {final_rating}/5
- Conceptual Support (Average): {intermediate_embedding_score:.3f}/1.0
- Semantic Cohesion (Average): {intermediate_gemini_semantic_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor about the *overall course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations for the *entire course*. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the *course's* pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Course Objective-Content Alignment: {final_rating}, Conceptual Support (Average): {intermediate_embedding_score:.3f}, and Semantic Cohesion (Average): {intermediate_gemini_semantic_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the overall objective-content alignment of the course. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths of the *entire course*. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        return user_prompt, instructor_prompt

    def prompt_module_learning_unity(self, intermediate_lo_coherence_score, intermediate_content_coherence_score, intermediate_name_alignment_score, module_name=None, final_rating=None):
        """
        Generates tailored prompts for the "Module Learning Unity" metric.
        Handles both module-level and course-level prompts.
        """
        if module_name:
            # Module-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

This reflection focuses on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the module's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Module Learning Unity (Score Range: 1-5):**
    * **Definition:** This metric evaluates the overall internal consistency and thematic coherence within a single module, ensuring all components work together seamlessly.
    * **Scoring:** 1 (disjointed components that hinder learning) to 5 (perfectly unified and cohesive learning experience).

* **Learning Objective Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This assesses how well the module's individual learning objectives relate to and support each other, forming a clear and unified set of goals.
    * **Scoring:** 0.0 (unrelated objectives) to 1.0 (highly coherent objectives).

* **Content Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures how well the various content items and topics within the module's syllabus relate to each other, ensuring a unified learning experience.
    * **Scoring:** 0.0 (disjointed content) to 1.0 (highly cohesive content).

* **Module Name Alignment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates how well the module's name accurately reflects and encompasses its learning objectives and syllabus content.
    * **Scoring:** 0.0 (misleading name) to 1.0 (perfectly representative name).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Module Learning Unity: {final_rating}/5
- Learning Objective Coherence: {intermediate_lo_coherence_score:.3f}/1.0
- Content Coherence: {intermediate_content_coherence_score:.3f}/1.0
- Module Name Alignment: {intermediate_name_alignment_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience.
* **Focus:** Perceived clarity, intellectual engagement, and relevance of content, all contributing to a unified learning experience within the module, based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on unity").

**Prompt Instruction:**
Given the underlying quality observations: Module Learning Unity: {final_rating}, Learning Objective Coherence: {intermediate_lo_coherence_score:.3f}, Content Coherence: {intermediate_content_coherence_score:.3f}, and Module Name Alignment: {intermediate_name_alignment_score:.3f}, write a short learner-style reflection on the overall internal unity of the module '{module_name}'. Simulate how a learner experienced the module's cohesiveness and focus. The last sentence should be a concise summary of the overall impression of the module's effectiveness from a learner's perspective, implicitly offering a key takeaway
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the module's syllabus by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall course structure. My aim is to pinpoint specific design strengths within the syllabus and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Module Learning Unity (Score Range: 1-5):**
    * **Definition:** This metric holistically evaluates the internal consistency and thematic coherence of all components within a module (learning objectives, syllabus content, module title).
    * **Scoring:** 1 (disjointed or contradictory components undermining learning) to 5 (exceptionally unified, cohesive, and clearly focused learning design).

* **Learning Objective Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This quantifies the internal semantic consistency among the module's learning objectives, ensuring they form a singular, focused set of instructional goals.
    * **Scoring:** 0.0 (disparate objectives) to 1.0 (highly unified and complementary objectives).

* **Content Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures the internal semantic consistency among the module's syllabus items, ensuring that all content contributes to a coherent thematic focus.
    * **Scoring:** 0.0 (disjointed content topics) to 1.0 (highly cohesive and interconnected content).

* **Module Name Alignment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates the precision with which the module's title accurately encapsulates and represents its learning objectives and syllabus content, ensuring immediate clarity of purpose.
    * **Scoring:** 0.0 (misleading or vague title) to 1.0 (perfectly representative and informative title).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Module Learning Unity: {final_rating}/5
- Learning Objective Coherence: {intermediate_lo_coherence_score:.3f}/1.0
- Content Coherence: {intermediate_content_coherence_score:.3f}/1.0
- Module Name Alignment: {intermediate_name_alignment_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Module Learning Unity: {final_rating}, Learning Objective Coherence: {intermediate_lo_coherence_score:.3f}, Content Coherence: {intermediate_content_coherence_score:.3f}, and Module Name Alignment: {intermediate_name_alignment_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the internal unity of the module '{module_name}'. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""
        else:
            # Course-level prompt
            user_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. My output should directly convey my feelings and observations about the course's effectiveness as a whole.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Course Learning Unity (Score Range: 1-5):**
    * **Definition:** This metric holistically evaluates the overall internal consistency and thematic coherence across all modules of the course, ensuring all components work together seamlessly.
    * **Scoring:** 1 (disjointed components that hinder learning) to 5 (perfectly unified and cohesive learning experience from start to finish).

* **Learning Objective Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This assesses how well the individual learning objectives across all modules relate to and support each other, forming a clear and unified set of course goals.
    * **Scoring:** 0.0 (unrelated objectives) to 1.0 (highly coherent objectives).

* **Content Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures how well the various content items and topics within all module syllabi relate to each other, ensuring a unified learning experience across the course.
    * **Scoring:** 0.0 (disjointed content) to 1.0 (highly cohesive content).

* **Module Name Alignment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates, on average, how well each module's name accurately reflects and encompasses its learning objectives and syllabus content.
    * **Scoring:** 0.0 (misleading names) to 1.0 (perfectly representative names).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Course Learning Unity: {final_rating}/5
- Learning Objective Coherence (Average): {intermediate_lo_coherence_score:.3f}/1.0
- Content Coherence (Average): {intermediate_content_coherence_score:.3f}/1.0
- Module Name Alignment (Average): {intermediate_name_alignment_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner, reflecting on the *entire course*.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing the personal experience of the *course's* unity.
* **Focus:** Perceived clarity, intellectual engagement, and relevance of content, all contributing to a unified learning experience across the course, based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on unity").

**Prompt Instruction:**
Given the underlying quality observations: Course Learning Unity: {final_rating}, Learning Objective Coherence (Average): {intermediate_lo_coherence_score:.3f}, Content Coherence (Average): {intermediate_content_coherence_score:.3f}, and Module Name Alignment (Average): {intermediate_name_alignment_score:.3f}, write a short learner-style reflection on the overall internal unity of the course. Simulate how a learner experienced the course's cohesiveness and focus from start to finish. The last sentence should be a concise summary of the overall impression of the course's effectiveness from a learner's perspective, implicitly offering a key takeaway
"""
            instructor_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on the *overall course's* design and its pedagogical effectiveness across all modules. Your insights aim to optimize the course for maximum learning impact and seamless integration of its components.

This assessment serves to validate the *entire course's* structure by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution of all modules to the overall course structure. The aim is to pinpoint specific design strengths within the overall course and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Course Learning Unity (Score Range: 1-5):**
    * **Definition:** This metric holistically evaluates the internal consistency and thematic coherence of all components across the entire course (learning objectives, syllabus content, module titles).
    * **Scoring:** 1 (disjointed or contradictory components undermining learning) to 5 (exceptionally unified, cohesive, and clearly focused learning design).

* **Learning Objective Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This quantifies the average internal semantic consistency among all module's learning objectives, ensuring they form a singular, focused set of instructional goals for the course.
    * **Scoring:** 0.0 (disparate objectives) to 1.0 (highly unified and complementary objectives).

* **Content Coherence (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This measures the average internal semantic consistency among all module's syllabus items, ensuring that all content contributes to a coherent thematic focus for the entire course.
    * **Scoring:** 0.0 (disjointed content topics) to 1.0 (highly cohesive and interconnected content).

* **Module Name Alignment (Intermediate Score Range: 0.0-1.0):**
    * **Definition:** This evaluates the average precision with which each module's title accurately encapsulates and represents its learning objectives and syllabus content across the course.
    * **Scoring:** 0.0 (misleading or vague titles) to 1.0 (perfectly representative and informative titles).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Course Learning Unity: {final_rating}/5
- Learning Objective Coherence (Average): {intermediate_lo_coherence_score:.3f}/1.0
- Content Coherence (Average): {intermediate_content_coherence_score:.3f}/1.0
- Module Name Alignment (Average): {intermediate_name_alignment_score:.3f}/1.0

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor about the *overall course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations for the *entire course*. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the *course's* pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Course Learning Unity: {final_rating}, Learning Objective Coherence (Average): {intermediate_lo_coherence_score:.3f}, Content Coherence (Average): {intermediate_content_coherence_score:.3f}, and Module Name Alignment (Average): {intermediate_name_alignment_score:.3f}, deliver a precise pedagogical assessment and concrete, actionable recommendations for the overall internal unity of the course. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths of the *entire course*. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
        return user_prompt, instructor_prompt


    def analyze_course_structure(self, course_metadata):
        """
        Complete course structure analysis
        Returns all five metrics on 1-5 scale along with their explanations,
        for both users and instructors.
        """
        modules = []
        course_info = {}

        for key, value in course_metadata.items():
            if key.startswith('Module'):
                module_data = {
                    'module_name': value.get('Name', f"Module {len(modules)+1}"),
                    'learning_objectives': value.get('Learning Objectives', []),
                    'syllabus': value.get('Syllabus', [])
                }
                modules.append(module_data)
            else:
                course_info[key] = value

        results = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                'metric1': executor.submit(self.metric1_inter_module_progression, modules),
                'metric2': executor.submit(self.metric2_course_module_alignment, course_info, modules),
                'metric3': executor.submit(self.metric3_intra_module_content_flow, modules),
                'metric4': executor.submit(self.metric4_objective_content_alignment, modules),
                'metric5': executor.submit(self.metric5_module_learning_unity, modules),
            }

            for name, future in futures.items():
                if name == 'metric1':
                    original_scores, intermediate_data = future.result()

                    for data in intermediate_data:
                        embedding_score = data['embedding_score']
                        gemini_score = data['gemini_semantic_score']
                        print(f"  - Embedding Score: {embedding_score:.2f}")
                        print(f"  - Gemini Semantic Score: {gemini_score:.2f}")

                    results['Inter Module Progression Coherence'] = {
                        "Module Transitions": []
                    }
                    if not modules or len(modules) < 2:
                        results['Inter Module Progression Coherence']['Module Transitions'].append({
                            "Detail": "Not enough modules to assess progression.",
                            "Learner Perspective Assessment": "There isn't enough course material to check how well modules connect.",
                            "Instructor Feedback": "Inter-module progression cannot be calculated with fewer than two modules."
                        })
                    else:
                        for i, score in enumerate(original_scores):
                            scaled_score = round(1 + 4 * float(score), 2)
                            module_pair_name = intermediate_data[i]['module_pair']

                            user_prompt_text, instructor_prompt_text = self.prompt_inter_module_progression_coherence(
                                intermediate_gemini_semantic_score=intermediate_data[i]['gemini_semantic_score'],
                                intermediate_embedding_score=intermediate_data[i]['embedding_score'],
                                module_pair=module_pair_name,
                                final_rating=scaled_score
                            )

                            user_exp = self.call_gemini(user_prompt_text) if self.gemini_model else "Explanation not available."
                            instructor_exp = self.call_gemini(instructor_prompt_text) if self.gemini_model else "Explanation not available."

                            results['Inter Module Progression Coherence']['Module Transitions'].append({
                                "From Module": module_pair_name.split(' to ')[0],
                                "To Module": module_pair_name.split(' to ')[1],
                                "Embedding Score": round(intermediate_data[i]['embedding_score'], 4),
                                "Gemini Semantic Score": round(intermediate_data[i]['gemini_semantic_score'], 4),
                                "Score": scaled_score,
                                "Learner Perspective Assessment": user_exp,
                                "Instructor Feedback": instructor_exp
                            })


                        # New Course-Level Summary for Metric 1
                        avg_score = float(np.mean(original_scores)) if original_scores else 0.0
                        avg_scaled_score = round(1 + 4 * avg_score, 2)
                        
                        avg_gemini_score = float(np.mean([d['gemini_semantic_score'] for d in intermediate_data])) if intermediate_data else 0.0
                        avg_embedding_score = float(np.mean([d['embedding_score'] for d in intermediate_data])) if intermediate_data else 0.0

                        user_prompt_text_course, instructor_prompt_text_course = self.prompt_inter_module_progression_coherence(
                            intermediate_gemini_semantic_score=avg_gemini_score,
                            intermediate_embedding_score=avg_embedding_score,
                            final_rating=avg_scaled_score,
                            module_pair=None # Indicate this is a course-level prompt
                        )
                        user_exp_course = self.call_gemini(user_prompt_text_course) if self.gemini_model else "Explanation not available."
                        instructor_exp_course = self.call_gemini(instructor_prompt_text_course) if self.gemini_model else "Explanation not available."

                        results['Inter Module Progression Coherence']['Course Level Evaluation'] = {
                            "Overall Score": avg_scaled_score,
                            "Learner Perspective Assessment": user_exp_course,
                            "Instructor Feedback": instructor_exp_course
                        }

                elif name == 'metric2':
                    # No changes for metric 2
                    original_scores, intermediate_data = future.result()
                    scaled_score = round(1 + 4 * float(original_scores), 2)
                    
                    results['Course Module Alignment'] = {
                        "Overall Score": scaled_score,
                    }

                    user_prompt_text, instructor_prompt_text = self.prompt_course_module_alignment(
                        intermediate_embedding_component_raw=float(intermediate_data["final_score_components"]["embedding_component"]) / 0.4,
                        intermediate_gemini_component_raw=float(intermediate_data["final_score_components"]["gemini_component"]) / 0.4,
                        intermediate_keyword_coverage_component_raw=float(intermediate_data["final_score_components"]["keyword_coverage_component"]) / 0.2,
                        final_rating=scaled_score
                    )

                    user_exp = self.call_gemini(user_prompt_text) if self.gemini_model else "Explanation not available."
                    instructor_exp = self.call_gemini(instructor_prompt_text) if self.gemini_model else "Explanation not available."

                    results['Course Module Alignment'].update({
                        "Learner Perspective Assessment": user_exp,
                        "Instructor Feedback": instructor_exp
                    })

                elif name == 'metric3':
                    original_scores, intermediate_data = future.result()
                    results['Intra Module Content Flow'] = {
                        "modules": []
                    }
                    if not modules:
                        results['Intra Module Content Flow']['modules'].append({
                            "Detail": "No modules to assess content flow.",
                            "Learner Perspective Assessment": "There are no modules to check content flow within.",
                            "Instructor Feedback": "Intra-module content flow cannot be calculated without modules."
                        })
                    else:
                        for i, score in enumerate(original_scores):
                            scaled_score = round(1 + 4 * float(score), 2)
                            module_name = modules[i].get('module_name', f"Module {i+1}")

                            user_prompt_text, instructor_prompt_text = self.prompt_intra_module_content_flow(
                                intermediate_sequential_similarities_avg=float(np.mean(intermediate_data[i]['sequential_similarities'])) if intermediate_data[i]['sequential_similarities'] else 0.0,
                                module_name=module_name,
                                final_rating=scaled_score
                            )

                            user_exp = self.call_gemini(user_prompt_text) if self.gemini_model else "Explanation not available."
                            instructor_exp = self.call_gemini(instructor_prompt_text) if self.gemini_model else "Explanation not available."

                            results['Intra Module Content Flow']['modules'].append({
                                "Module Name": module_name,
                                "Average Sequential Similarity": round(np.mean(intermediate_data[i]['sequential_similarities']) if intermediate_data[i]['sequential_similarities'] else 0.0, 4),
                                "Score": scaled_score,
                                "Learner Perspective Assessment": user_exp,
                                "Instructor Feedback": instructor_exp
                            })


                        # New Course-Level Summary for Metric 3
                        avg_score = float(np.mean(original_scores)) if original_scores else 0.0
                        avg_scaled_score = round(1 + 4 * avg_score, 2)
                        
                        avg_sequential_similarity = float(np.mean([np.mean(d['sequential_similarities']) for d in intermediate_data if d['sequential_similarities']])) if any(d['sequential_similarities'] for d in intermediate_data) else 0.0

                        user_prompt_text_course, instructor_prompt_text_course = self.prompt_intra_module_content_flow(
                            intermediate_sequential_similarities_avg=avg_sequential_similarity,
                            module_name=None, # Indicate this is a course-level prompt
                            final_rating=avg_scaled_score
                        )
                        user_exp_course = self.call_gemini(user_prompt_text_course) if self.gemini_model else "Explanation not available."
                        instructor_exp_course = self.call_gemini(instructor_prompt_text_course) if self.gemini_model else "Explanation not available."

                        results['Intra Module Content Flow']['Course Level Evaluation'] = {
                            "Overall Score": avg_scaled_score,
                            "Learner Perspective Assessment": user_exp_course,
                            "Instructor Feedback": instructor_exp_course
                        }

                elif name == 'metric4':
                    original_scores, intermediate_data = future.result()
                    results['Objective Content Alignment'] = {
                        "modules": []
                    }
                    if not modules:
                         results['Objective Content Alignment']['modules'].append({
                            "Detail": "No modules to assess objective-content alignment.",
                            "Learner Perspective Assessment": "There are no modules to check how content aligns with learning goals.",
                            "Instructor Feedback": "Objective-content alignment cannot be calculated without modules."
                        })
                    else:
                        for i, score in enumerate(original_scores):
                            scaled_score = round(1 + 4 * float(score), 2)
                            module_name = modules[i].get('module_name', f"Module {i+1}")

                            user_prompt_text, instructor_prompt_text = self.prompt_objective_content_alignment(
                                intermediate_embedding_score=intermediate_data[i]['embedding_score'],
                                intermediate_gemini_semantic_score=intermediate_data[i]['gemini_semantic_score'],
                                module_name=module_name,
                                final_rating=scaled_score
                            )

                            user_exp = self.call_gemini(user_prompt_text) if self.gemini_model else "Explanation not available."
                            instructor_exp = self.call_gemini(instructor_prompt_text) if self.gemini_model else "Explanation not available."

                            results['Objective Content Alignment']['modules'].append({
                                "Module Name": module_name,
                                "Score": scaled_score,
                                "Embedding Score": round(intermediate_data[i]['embedding_score'], 4),
                                "Gemini Semantic Score": round(intermediate_data[i]['gemini_semantic_score'], 4),
                                "Score": scaled_score,
                                "Learner Perspective Assessment": user_exp,
                                "Instructor Feedback": instructor_exp,
                            })


                        # New Course-Level Summary for Metric 4
                        avg_score = float(np.mean(original_scores)) if original_scores else 0.0
                        avg_scaled_score = round(1 + 4 * avg_score, 2)
                        
                        avg_embedding_score = float(np.mean([d['embedding_score'] for d in intermediate_data])) if intermediate_data else 0.0
                        avg_gemini_score = float(np.mean([d['gemini_semantic_score'] for d in intermediate_data])) if intermediate_data else 0.0
                        
                        user_prompt_text_course, instructor_prompt_text_course = self.prompt_objective_content_alignment(
                            intermediate_embedding_score=avg_embedding_score,
                            intermediate_gemini_semantic_score=avg_gemini_score,
                            module_name=None, # Indicate this is a course-level prompt
                            final_rating=avg_scaled_score
                        )
                        user_exp_course = self.call_gemini(user_prompt_text_course) if self.gemini_model else "Explanation not available."
                        instructor_exp_course = self.call_gemini(instructor_prompt_text_course) if self.gemini_model else "Explanation not available."

                        results['Objective Content Alignment']['Course Level Evaluation'] = {
                            "Overall Score": avg_scaled_score,
                            "Learner Perspective Assessment": user_exp_course,
                            "Instructor Feedback": instructor_exp_course
                        }
                    
                elif name == 'metric5':
                    original_scores, intermediate_data = future.result()
                    results['Module Learning Unity'] = {
                        "modules": []
                    }
                    if not modules:
                        results['Module Learning Unity']['modules'].append({
                            "Detail": "No modules to assess learning unity.",
                            "Learner Perspective Assessment": "There are no modules to check for internal consistency.",
                            "Instructor Feedback": "Module learning unity cannot be calculated without modules."
                        })
                    else:
                        for i, score in enumerate(original_scores):
                            scaled_score = round(1 + 4 * float(score), 2)
                            module_name = modules[i].get('module_name', f"Module {i+1}")

                            user_prompt_text, instructor_prompt_text = self.prompt_module_learning_unity(
                                intermediate_lo_coherence_score=intermediate_data[i]['lo_coherence_score'],
                                intermediate_content_coherence_score=intermediate_data[i]['content_coherence_score'],
                                intermediate_name_alignment_score=intermediate_data[i]['name_alignment_score'],
                                module_name=module_name,
                                final_rating=scaled_score
                            )

                            user_exp = self.call_gemini(user_prompt_text) if self.gemini_model else "Explanation not available."
                            instructor_exp = self.call_gemini(instructor_prompt_text) if self.gemini_model else "Explanation not available."

                            results['Module Learning Unity']['modules'].append({
                                "Module Name": module_name,
                                "Learning Objective Coherence": round(intermediate_data[i]['lo_coherence_score'], 4),
                                "Content Coherence": round(intermediate_data[i]['content_coherence_score'], 4),
                                "Module Name Alignment": round(intermediate_data[i]['name_alignment_score'], 4),
                                "Score": scaled_score,
                                "Learner Perspective Assessment": user_exp,
                                "Instructor Feedback": instructor_exp,
                            })

                        # New Course-Level Summary for Metric 5
                        avg_score = float(np.mean(original_scores)) if original_scores else 0.0
                        avg_scaled_score = round(1 + 4 * avg_score, 2)
                        
                        avg_lo_coherence = float(np.mean([d['lo_coherence_score'] for d in intermediate_data])) if intermediate_data else 0.0
                        avg_content_coherence = float(np.mean([d['content_coherence_score'] for d in intermediate_data])) if intermediate_data else 0.0
                        avg_name_alignment = float(np.mean([d['name_alignment_score'] for d in intermediate_data])) if intermediate_data else 0.0

                        user_prompt_text_course, instructor_prompt_text_course = self.prompt_module_learning_unity(
                            intermediate_lo_coherence_score=avg_lo_coherence,
                            intermediate_content_coherence_score=avg_content_coherence,
                            intermediate_name_alignment_score=avg_name_alignment,
                            module_name=None, # Indicate this is a course-level prompt
                            final_rating=avg_scaled_score
                        )
                        user_exp_course = self.call_gemini(user_prompt_text_course) if self.gemini_model else "Explanation not available."
                        instructor_exp_course = self.call_gemini(instructor_prompt_text_course) if self.gemini_model else "Explanation not available."

                        results['Module Learning Unity']['Course Level Evaluation'] = {
                            "Overall Score": avg_scaled_score,
                            "Learner Perspective Assessment": user_exp_course,
                            "Instructor Feedback": instructor_exp_course
                        }

        return results
    
# --------- New Usage Function for GDrive Integration ---------

def analyze_course_from_gdrive(service_account_file, folder_id, gemini_api_key):
    service = get_gdrive_service(service_account_file)
    metadata = fetch_metadata_json(service, folder_id)
    if metadata is None:
        print("No metadata found. Aborting analysis.")
        return None
    analyzer = CourseStructureAnalyzer(gemini_api_key=gemini_api_key)
    results = analyzer.analyze_course_structure(metadata)
    return results


# Sample usage
if __name__ == "__main__":
    # Ensure you have your service account file and Gemini API key configured
    # For local testing, replace with your actual values
    service_account_file = r""
    folder_id = "" # Replace with your folder ID
    gemini_api_key = "" # Replace with your actual Gemini API Key

    start_time = time.time()

    results = analyze_course_from_gdrive(service_account_file, folder_id, gemini_api_key)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if results:
        print(json.dumps(results, indent=4))

        output_filename = "LSA_report.json"
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nAnalysis results saved to {output_filename}")
    else:
        print("Course analysis failed or no metadata found.")

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
