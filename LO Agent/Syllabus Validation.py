import json
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Google Drive and API imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai

# NLP and text processing imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import spacy
import textstat # Make sure to install: pip install textstat

# Download required NLTK data (only if still used)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger') # For POS tagging
except LookupError:
    print("Downloading NLTK data. This may take a moment...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger') # For POS tagging
    print("NLTK data download complete.")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded.")

class SyllabusValidator:
    def __init__(self, drive_service: Any, gemini_api_key: str):
        """
        Initialize the Syllabus Validator with Google Drive and Gemini AI services.
        Args:
            drive_service: An authenticated Google Drive service object.
            gemini_api_key: Gemini AI API key
        """
        self.drive_service = drive_service
        # Configure Gemini AI
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Define expected ranges for cognitive_complexity and topic_coverage based on course level
        self.expected_ranges = {
            "Beginner": {
                "bloom_avg": (1.0, 2.0),
                "unique_topics": (5, 15),
                "readability_grade": (5.0, 8.0), # Example range for beginner
                "diversity_ratio": (0.3, 0.6) # Example range for beginner
            },
            "Intermediate": {
                "bloom_avg": (2.0, 4.0),
                "unique_topics": (15, 30),
                "readability_grade": (8.0, 12.0), # Example range for intermediate
                "diversity_ratio": (0.5, 0.8) # Example range for intermediate
            },
            "Advanced": {
                "bloom_avg": (4.0, 6.0),
                "unique_topics": (30, 50), # Assuming 50+ is also fine for advanced
                "readability_grade": (12.0, 16.0), # Example range for advanced
                "diversity_ratio": (0.7, 1.0) # Example range for advanced
            }
        }

    def get_json_content(self, file_id: str) -> Dict:
        """Get and parse JSON file content."""
        request = self.drive_service.files().get_media(fileId=file_id)
        file_content = request.execute()
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')
        return json.loads(file_content)

    def fetch_metadata_json(self, folder_id: str) -> Optional[Dict]:
        """
        Find and fetch the metadata.json file in the specified folder.
        Modified to find any JSON file and assume it's the metadata.
        For robust usage, you might want to specify the filename explicitly.
        """
        query = f"'{folder_id}' in parents and mimeType='application/json' and trashed=false"
        results = self.drive_service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        files = results.get('files', [])
        if not files:
            print(f"No JSON file found in folder {folder_id}")
            return None
        # Assuming the first JSON file found is the one we want.
        # If there are multiple, you might need a more specific name.
        metadata_file_id = files[0]['id']
        print(f"Found JSON file '{files[0]['name']}' with ID: {metadata_file_id}")
        try:
            metadata_content = self.get_json_content(metadata_file_id)
            return metadata_content
        except Exception as e:
            print(f"Error fetching JSON file content: {str(e)}")
            return None


    def get_gemini_objective_syllabus_alignment_score(self, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the alignment score between module learning objectives and syllabus content,
        using an enhanced set of criteria for Gemini's evaluation.
        """
        module_learning_objectives = module_data.get('Learning Objectives', [])
        module_syllabus = module_data.get('Syllabus', [])
        module_title = module_data.get('Module Title', module_data.get('Name', 'Untitled Module'))

        prompt = f"""
As an expert in educational course design, meticulously evaluate the pedagogical alignment between the module's learning objectives and its syllabus content. The evaluation should assess how effectively the syllabus content directly supports and enables the achievement of each stated learning objective.

Consider the following critical aspects for your granular assessment of alignment:

-   **Objective Coverage & Completeness:**
    * Are ALL learning objectives explicitly and comprehensively addressed within the syllabus content?
    * Identify any objectives that are entirely missing content, or where content is clearly insufficient to achieve the stated outcome.
    * Conversely, identify any content elements in the syllabus that do not clearly map to any stated learning objective (extraneous material).
-   **Depth of Treatment & Cognitive Level Alignment:**
    * Does the syllabus content provide the necessary depth and complexity to enable learners to achieve the *stated cognitive level* of each objective (e.g., "analyze" requires more depth than "recall")?
    * Pinpoint instances where content is too superficial for a higher-order objective, or excessively detailed for a foundational one.
-   **Relevance & Focus:**
    * Is every component of the syllabus content (topics, readings, activities) directly relevant and necessary for achieving the module's learning objectives?
    * Flag any syllabus items that appear tangential, redundant, or unrelated to the learning objectives.
-   **Clarity of Connection & Learnability:**
    * How easily can a typical learner discern the direct connection between each learning objective and the specific syllabus components designed to help them achieve it?
    * Note any ambiguous links or situations where the connection is obscure, requiring the learner to infer heavily.
-   **Sequence & Scaffolded Learning (if applicable):**
    * Does the order of topics within the syllabus logically scaffold the learning, progressing from foundational concepts to more advanced ones, in a way that supports the objectives' achievement?
    * Identify any illogical sequencing that might hinder objective mastery.

Module Title: {module_title}
Module Learning Objectives:\n{chr(10).join(module_learning_objectives)}
Module Content (Syllabus):\n{chr(10).join(module_syllabus)}

Based on this evaluation, provide the following in strict JSON format:

1.  **alignment_score (1.0-5.0):** A float representing the overall pedagogical effectiveness of the syllabus in achieving the learning objectives.
    * *Self-calibration for score:*
        * 1.0-2.0: Poor alignment; significant gaps, irrelevance, or insufficient depth for stated cognitive levels.
        * 2.1-3.0: Moderate alignment; some objectives partially met, noticeable gaps, extraneous content, or cognitive level mismatches.
        * 3.1-4.0: Good alignment; most objectives well-covered with appropriate depth, minor areas for improvement in relevance or clarity.
        * 4.1-5.0: Excellent alignment; all objectives fully supported with appropriate depth for cognitive levels, highly relevant, clear connections, and logical sequencing.
2.  **analysis (string):** A concise analysis (max ~150 words) detailing the alignment's strengths and weaknesses. Crucially, **cite specific examples or key areas** from the provided objectives and syllabus content to support your points.
3.  **recommendation (string):** One highly specific and actionable step to enhance the alignment. Clearly state *what* needs to be done (e.g., "Add a section on X to Module Y") and *why* this specific action will effectively improve the alignment with a particular objective or set of objectives.

Respond in strict JSON:
{{
    "alignment_score": <float>,
    "analysis": "<string>",
    "recommendation": "<string>"
}}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()

            try:
                gemini_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error from Gemini (objective-syllabus alignment): {e}")
                print(f"Raw Gemini Response (attempting parse): {response_text}")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        gemini_result = json.loads(json_match.group(0))
                        print("Successfully parsed JSON from messy text for objective-syllabus alignment.")
                    except json.JSONDecodeError:
                        print("Failed to parse JSON even from regex match for objective-syllabus alignment. Returning fallback.")
                        raise

            gemini_result['alignment_score'] = round(float(gemini_result['alignment_score']), 2)
            return {
                'alignment_score': gemini_result['alignment_score'],
                'analysis': gemini_result['analysis'],
                'recommendation': gemini_result['recommendation'],
                'status': 'success'
            }
        except Exception as e:
            print(f"Error calling Gemini for objective-syllabus alignment or parsing its response: {e}")
            return {
                'alignment_score': 1.0, # Fallback to lowest score
                'analysis': "Fallback: Could not get Gemini alignment analysis for this module.",
                'recommendation': "Ensure Gemini API is configured and input data is valid for objective-syllabus alignment.",
                'status': 'fallback'
            }

    def get_gemini_course_syllabus_alignment_score(self, course_data: Dict[str, Any], module_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the alignment score between the overall course description and the module's syllabus content,
        using an enhanced set of criteria for Gemini's evaluation.
        """
        course_description = course_data.get('About this Course', 'No course description provided.')
        module_syllabus = module_data.get('Syllabus', [])
        module_title = module_data.get('Module Title', module_data.get('Name', 'Untitled Module'))

        prompt = f"""
As an expert in educational course design, meticulously evaluate the strategic alignment between the overall course description and this specific module's syllabus content. The evaluation should assess how well the module's syllabus content integrates with and contributes to the broader learning goals and thematic progression outlined in the course description.

Consider the following critical aspects for your granular assessment of strategic alignment:

-   **Thematic Integration & Overall Cohesion:**
    * Does this module's syllabus content seamlessly integrate with and consistently reflect the main themes, overarching goals, and intellectual narrative introduced in the overall course description?
    * Identify any thematic inconsistencies, significant omissions of core course themes, or introduction of entirely new, unaligned themes within this module.
-   **Progressive Contribution & Learning Path:**
    * Does the content of this module logically advance the learner's journey as described in the overall course flow (e.g., building on previous modules, preparing for subsequent ones)?
    * Pinpoint instances where the module's content appears out of sequence, introduces concepts prematurely, or retreads ground already covered extensively elsewhere in the course description.
-   **Scope & Balance within Course Context:**
    * Is the scope and depth of this module's content appropriately balanced relative to its role within the entire course?
    * Identify if the module is disproportionately broad (e.g., attempting to cover too much for a single module, duplicating other modules) or too narrow (e.g., failing to contribute sufficiently to broader course outcomes).
-   **Direct Contribution to Course-Level Outcomes:**
    * How explicitly and substantially does this module's syllabus content contribute to the achievement of the overall intended learning outcomes and competencies of the *entire course*?
    * Highlight specific aspects of the module that directly advance course outcomes, and conversely, any course outcomes that this module, by its content, fails to adequately support.
-   **Relevance to Target Audience & Prerequisites (implied by Course Description):**
    * Does the module's content assume appropriate prior knowledge or prepare learners for subsequent material, as suggested by the course description's target audience or prerequisites?
    * Note any mismatches where the module's content seems too advanced or too remedial for its assumed place in the course.

Course Description: {course_description}
Module Title: {module_title}
Module Content (Syllabus):\n{chr(10).join(module_syllabus)}

Based on this evaluation, provide the following in strict JSON format:

1.  **alignment_score (1.0-5.0):** A float representing how well this module's syllabus content aligns with and contributes to the overall course description and objectives.
    * *Self-calibration for score:*
        * 1.0-2.0: Poor alignment; module content largely disconnected or contradictory to course description.
        * 2.1-3.0: Moderate alignment; some connection, but significant thematic gaps, redundancy, or awkward progression.
        * 3.1-4.0: Good alignment; module generally supports course goals, minor areas for refinement in integration.
        * 4.1-5.0: Excellent alignment; module content seamlessly integrates, significantly contributes to, and is perfectly scoped within the overall course.
2.  **analysis (string):** A concise analysis (max ~150 words) detailing the alignment's strengths and weaknesses. Crucially, **cite specific examples or key areas** from the provided course description and module syllabus content to support your points.
3.  **recommendation (string):** One highly specific and actionable step to enhance the alignment. Clearly state *what* needs to be done (e.g., "Adjust Module X to cover Y before Z") and *why* this specific action will strengthen the module's contribution to the overall course objectives.

Respond in strict JSON:
{{
    "alignment_score": <float>,
    "analysis": "<string>",
    "recommendation": "<string>"
}}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()

            try:
                gemini_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error from Gemini (course-syllabus alignment): {e}")
                print(f"Raw Gemini Response (attempting parse): {response_text}")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        gemini_result = json.loads(json_match.group(0))
                        print("Successfully parsed JSON from messy text for course-syllabus alignment.")
                    except json.JSONDecodeError:
                        print("Failed to parse JSON even from regex match for course-syllabus alignment. Returning fallback.")
                        raise

            gemini_result['alignment_score'] = round(float(gemini_result['alignment_score']), 2)
            return {
                'alignment_score': gemini_result['alignment_score'],
                'analysis': gemini_result['analysis'],
                'recommendation': gemini_result['recommendation'],
                'status': 'success'
            }
        except Exception as e:
            print(f"Error calling Gemini for course-syllabus alignment or parsing its response: {e}")
            return {
                'alignment_score': 1.0, # Fallback to lowest score
                'analysis': "Fallback: Could not get Gemini alignment analysis for this module.",
                'recommendation': "Ensure Gemini API is configured and input data is valid for course-syllabus alignment.",
                'status': 'fallback'
            }
        
    def get_gemini_bloom_classification(self, learning_objective: str) -> Dict[str, Any]:
        """
        Classifies the Bloom's Taxonomy level(s) of a learning objective using Gemini AI.
        """
        prompt = f"""
As an expert in educational pedagogy and Bloom's Taxonomy, classify the following learning objective according to Bloom's Taxonomy cognitive levels (1: Remember, 2: Understand, 3: Apply, 4: Analyze, 5: Evaluate, 6: Create).
If multiple levels are applicable, list all of them. Focus on the highest cognitive process demanded by the objective.

Learning Objective: "{learning_objective}"

Provide:
1. Bloom's Level(s): A list of integer(s) representing the most appropriate Bloom's Taxonomy level(s).
2. Justification: A brief explanation for your classification, referencing the action verbs and implied cognitive processes.

Respond in strict JSON:
{{
    "bloom_levels": [<integer>, ...],
    "justification": "<string>"
}}
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            # Robust JSON extraction
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()

            try:
                gemini_result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error from Gemini (Bloom's): {e}")
                print(f"Raw Gemini Response (attempting parse): {response_text}")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        gemini_result = json.loads(json_match.group(0))
                        print("Successfully parsed JSON from messy text for Bloom's.")
                    except json.JSONDecodeError:
                        print("Failed to parse JSON even from regex match for Bloom's. Returning fallback.")
                        raise

            # Ensure bloom_levels is a list of integers
            bloom_levels = [int(level) for level in gemini_result.get('bloom_levels', [1])]
            return {
                'bloom_levels': sorted(list(set(bloom_levels))), # Ensure unique and sorted
                'justification': gemini_result.get('justification', 'No justification provided.'),
                'status': 'success'
            }
        except Exception as e:
            print(f"Error calling Gemini for Bloom's classification or parsing its response: {e}")
            return {
                'bloom_levels': [1], # Fallback to lowest level
                'justification': "Fallback: Could not classify Bloom's level via Gemini.",
                'status': 'fallback'
            }

    def calculate_cognitive_complexity_and_topic_coverage(self, module_data: Dict[str, Any], course_level: str, learning_objectives_bloom: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates cognitive complexity and topic coverage scores for a single module based on Bloom's Taxonomy,
        unique topics, topic diversity, and readability.
        """
        module_learning_objectives = module_data.get('Learning Objectives', [])
        module_syllabus = module_data.get('Syllabus', [])

        # --- Cognitive Complexity Calculation: Average Bloom level ---
        all_bloom_levels = []
        for lo_data in learning_objectives_bloom:
            levels = lo_data['bloom_classification'].get('bloom_levels', [1])
            all_bloom_levels.extend(levels)

        average_bloom_level = sum(all_bloom_levels) / len(all_bloom_levels) if all_bloom_levels else 1.0
        average_bloom_level = round(average_bloom_level, 2)

        # --- Combined text for spaCy and readability ---
        combined_text = " ".join(module_learning_objectives + module_syllabus)
        doc = nlp(combined_text)

        # --- Topic Coverage Calculation: Unique Topics & Diversity Ratio ---
        lemmatized_nouns = [
            self.lemmatizer.lemmatize(token.text.lower())
            for token in doc if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha and len(token) > 2
        ]
        unique_lemmatized_nouns = set(lemmatized_nouns)
        total_nouns_count = len(lemmatized_nouns)

        diversity_ratio = len(unique_lemmatized_nouns) / total_nouns_count if total_nouns_count > 0 else 0.0
        diversity_ratio = round(diversity_ratio, 2)

        # --- Cognitive Complexity Calculation: Readability Score (Flesch-Kincaid Grade) ---
        readability_score = textstat.flesch_kincaid_grade(combined_text) if combined_text else 0.0
        readability_score = round(readability_score, 2)

        # --- TF-IDF Proxy for Complexity (conceptual, using average word/sentence length) ---
        words = [token.text for token in doc if token.is_alpha]
        sentences = [sent.text for sent in doc.sents]

        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0

        # Combine these into a conceptual complexity score. This weighting is illustrative.
        # Higher avg_word_length and avg_sentence_length could indicate more complexity.
        conceptual_complexity_score = (avg_word_length / 5.0) + (avg_sentence_length / 20.0) # Normalize to a reasonable range
        conceptual_complexity_score = min(5.0, max(0.0, conceptual_complexity_score)) # Clamp
        conceptual_complexity_score = round(conceptual_complexity_score, 2)

        # --- Scoring based on Expected Ranges ---
        target_level_ranges = self.expected_ranges.get(course_level, self.expected_ranges["Intermediate"])
        expected_bloom_min, expected_bloom_max = target_level_ranges["bloom_avg"]
        expected_topics_min, expected_topics_max = target_level_ranges["unique_topics"]
        expected_read_min, expected_read_max = target_level_ranges["readability_grade"]
        expected_diversity_min, expected_diversity_max = target_level_ranges["diversity_ratio"]

        # Cognitive Complexity Score (combining Bloom's and complexity indicators)
        cognitive_complexity_score_bloom_component = 0
        if average_bloom_level >= expected_bloom_min and average_bloom_level <= expected_bloom_max:
            cognitive_complexity_score_bloom_component = 5
        elif average_bloom_level < expected_bloom_min:
            cognitive_complexity_score_bloom_component = max(1, 5 - math.ceil((expected_bloom_min - average_bloom_level) * 1.5))
        else:
            cognitive_complexity_score_bloom_component = max(1, 5 - math.ceil((average_bloom_level - expected_bloom_max) * 0.5))

        cognitive_complexity_score_readability_component = 0
        if readability_score >= expected_read_min and readability_score <= expected_read_max:
            cognitive_complexity_score_readability_component = 5
        elif readability_score < expected_read_min:
            cognitive_complexity_score_readability_component = max(1, 5 - math.ceil((expected_read_min - readability_score) * 1))
        else:
            cognitive_complexity_score_readability_component = max(1, 5 - math.ceil((readability_score - expected_read_max) * 0.5))

        # Overall cognitive complexity score: weighted average of components
        cognitive_complexity_score = round((cognitive_complexity_score_bloom_component * 0.6 + cognitive_complexity_score_readability_component * 0.4), 2)
        cognitive_complexity_score = max(1.0, min(5.0, cognitive_complexity_score))

        # Topic Coverage Score (combining unique topics and diversity ratio)
        topic_coverage_score_topics_component = 0
        if len(unique_lemmatized_nouns) >= expected_topics_min and len(unique_lemmatized_nouns) <= expected_topics_max:
            topic_coverage_score_topics_component = 5
        elif len(unique_lemmatized_nouns) < expected_topics_min:
            topic_coverage_score_topics_component = max(1, 5 - math.ceil((expected_topics_min - len(unique_lemmatized_nouns)) / 5))
        else:
            topic_coverage_score_topics_component = max(1, 5 - math.ceil((len(unique_lemmatized_nouns) - expected_topics_max) / 10))

        topic_coverage_score_diversity_component = 0
        if diversity_ratio >= expected_diversity_min and diversity_ratio <= expected_diversity_max:
            topic_coverage_score_diversity_component = 5
        elif diversity_ratio < expected_diversity_min:
            topic_coverage_score_diversity_component = max(1, 5 - math.ceil((expected_diversity_min - diversity_ratio) * 10))
        else:
            topic_coverage_score_diversity_component = max(1, 5 - math.ceil((diversity_ratio - expected_diversity_max) * 5))

        # Overall topic coverage score: weighted average
        topic_coverage_score = round((topic_coverage_score_topics_component * 0.7 + topic_coverage_score_diversity_component * 0.3), 2)
        topic_coverage_score = max(1.0, min(5.0, topic_coverage_score))

        analysis = (
            f"Cognitive complexity is assessed by the average Bloom's Taxonomy level of learning objectives "
            f"(Current: {average_bloom_level}, Expected for {course_level}: {expected_bloom_min}-{expected_bloom_max}) "
            f"and readability (Flesch-Kincaid Grade: {readability_score}, Expected: {expected_read_min}-{expected_read_max}). "
            f"Topic coverage is assessed by the number of unique topics (lemmatized nouns: {len(unique_lemmatized_nouns)}, "
            f"Expected for {course_level}: {expected_topics_min}-{expected_topics_max}) "
            f"and topic diversity (Ratio: {diversity_ratio:.2f}, Expected: {expected_diversity_min}-{expected_diversity_max})."
        )
        recommendation = (
            f"To improve cognitive complexity, adjust learning objectives to target appropriate Bloom's levels and ensure content "
            f"is written at a suitable readability for a {course_level} course. "
            f"To improve topic coverage, ensure a sufficient number of distinct topics are introduced, and that topic selection "
            f"demonstrates appropriate diversity."
        )

        return {
            'cognitive_complexity_score': cognitive_complexity_score,
            'topic_coverage_score': topic_coverage_score,
            'average_bloom_level': average_bloom_level,
            'num_unique_topics': len(unique_lemmatized_nouns),
            'topic_diversity_ratio': diversity_ratio,
            'readability_score': readability_score,
            'conceptual_complexity_proxy': conceptual_complexity_score,
            'analysis': analysis,
            'recommendation': recommendation
        }

    def _generate_final_score_explanations_with_gemini(self, objective_syllabus_alignment_score: float, course_syllabus_alignment_score: float, cognitive_complexity_score: float, topic_coverage_score: float, final_rating: float) -> Dict[str, Any]:
        """
        Generates tailored user and instructor explanations for the final module score.
        This function uses the provided prompt structure for detailed and specific output.
        """
        # User Perspective Assessment Prompt
        user_prompt_template = f"""
Persona: You are simulating a learner's direct experience and perception of a module's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the module.

As a learner, reflect on your module experience, focusing on how effectively the module's interconnected elements (goals, content, intellectual demands) came together to deliver a cohesive, accessible, and ultimately enriching learning experience.

**Evaluation Methodology (How my experience is being assessed behind the scenes):**

* **Objective-Syllabus Alignment (Score Range: 1-5):**
    * **Definition:** This metric indicates how precisely the module's stated learning objectives directly corresponded to, and were comprehensively covered by, the actual content provided.
    * **Scoring:** 1 (significant disconnects or content deficiencies) to 5 (perfect congruence and comprehensive delivery).

* **Course-Syllabus Alignment (Score Range: 1-5):**
    * **Definition:** This metric assesses how well this module seamlessly fit into the overall course flow, building on previous knowledge and preparing for subsequent topics.
    * **Scoring:** 1 (module content was largely disconnected or contradictory) to 5 (seamless integration and strategic contribution).

* **Cognitive Complexity (Score Range: 1-5):**
    * **Definition:** This metric gauges the module's cognitive demands and intellectual rigor, ensuring it was appropriately stimulating.
    * **Scoring:** 1 (content was overly simplistic or lacked rigor) to 5 (optimally stimulating, intellectually rigorous, and appropriately paced).

* **Topic Coverage (Score Range: 1-5):**
    * **Definition:** This metric quantifies the breadth, variety, and comprehensiveness of topics covered within the module.
    * **Scoring:** 1 (very limited topics or missing key concepts) to 5 (broad, diverse, and well-structured coverage of all essential subject matter).

* **Final Module Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the overall aggregated score for the module's effectiveness, reflecting how well all elements coalesced for my learning experience.
    * **Scoring:** 1 (significant impediment to effective learning) to 5 (an outstanding and highly effective learning experience).

**Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):**
- Objective-Syllabus Alignment: {objective_syllabus_alignment_score}/5
- Course-Syllabus Alignment: {course_syllabus_alignment_score}/5
- Cognitive Complexity: {cognitive_complexity_score}/5
- Topic Coverage: {topic_coverage_score}/5
- Final Rating: {final_rating}/5

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 2–4 sentences, describing my personal experience. Avoid passive voice where possible; use "I felt," "I found," "my understanding."
* **Focus:** My perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the module scored high on alignment").

**Prompt Instruction:**
Given the underlying quality observations: Objective-Syllabus Alignment: {objective_syllabus_alignment_score}, Course-Syllabus Alignment: {course_syllabus_alignment_score}, Cognitive Complexity: {cognitive_complexity_score}, Topic Coverage: {topic_coverage_score}, and a Final Rating: {final_rating}, write a short learner-style reflection on the module's quality. Simulate how a learner, experienced engaging with the module's content and structure. The last sentence should clearly imply the overall experience of the learner.
"""
        
        # Instructor Feedback Prompt
        instructor_prompt_template = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a module's design and its overall pedagogical effectiveness. Your insights aim to optimize the module for maximum learning impact and seamless integration within the broader curriculum.

As a pedagogical consultant, you are tasked with providing feedback on a module's syllabus to validate its design and effectiveness. This assessment aims to pinpoint design strengths and identify actionable areas for improvement.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Learning Objective-Syllabus Alignment (Score Range: 1-5):**
    * **Definition:** This metric quantifies how precisely the module's stated learning objectives directly correspond to, and are comprehensively covered by, the actual content presented in the syllabus. It ensures that what is promised to be learned is clearly and fully delivered.
    * **Scoring:** 1 (significant disconnects or content deficiencies relative to objectives) to 5 (perfect congruence; all objectives are fully supported, clearly addressed, and content is optimally aligned).

* **Course-Syllabus Alignment (Score Range: 1-5):**
    * **Definition:** This metric rigorously assesses how coherently and strategically this specific module's content integrates with the broader themes, overarching goals, and progressive narrative of the entire course. It confirms the module's meaningful contribution to the larger curriculum's flow.
    * **Scoring:** 1 (module content is largely disconnected or contradictory to the overall course direction) to 5 (module content seamlessly integrates, significantly contributes to, and is perfectly scoped within the overall course framework).

* **Cognitive Complexity (Score Range: 1-5):**
    * **Definition:** This metric evaluates the intellectual depth and challenge the module offers, ensuring it is appropriately rigorous for the expected academic level and student learning progression. It considers the cognitive demands implied by objectives and the readability of materials.
    * **Scoring:** 1 (content is overly simplistic or lacks intellectual rigor for the expected level) to 5 (optimally stimulating, intellectually rigorous, and appropriately paced for advanced learning).

* **Topic Coverage (Score Range: 1-5):
    * **Definition:** This metric measures the breadth, variety, and comprehensiveness of the distinct topics presented within the module. It assures that a sufficient range of concepts is introduced and explored with adequate diversity to build a robust understanding.
    * **Scoring:** 1 (very limited topics; narrow scope, missing key concepts, or insufficient detail) to 5 (very comprehensive topics; broad, diverse, and well-structured coverage of all essential subject matter).

* **Final Module Quality Rating (Overall Score Range: 1-5):**
    * **Definition:** This is the holistic and quantifiable assessment of the module's overall instructional design excellence, reflecting its internal consistency, fidelity to learning outcomes, and strategic contribution to the course.
    * **Scoring:** 1 (poor design; indicative of fundamental pedagogical flaws requiring urgent redesign) to 5 (excellent design; a model of instructional excellence, demonstrating outstanding clarity, coherence, and effectiveness).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Objective-Syllabus Alignment: {objective_syllabus_alignment_score}/5
- Course-Syllabus Alignment: {course_syllabus_alignment_score}/5
- Cognitive Complexity: {cognitive_complexity_score}/5
- Topic Coverage: {topic_coverage_score}/5
- Final Rating: {final_rating}/5

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the module's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Objective-Syllabus Alignment: {objective_syllabus_alignment_score}, Course-Syllabus Alignment: {course_syllabus_alignment_score}, Cognitive Complexity: {cognitive_complexity_score}, Topic Coverage: {topic_coverage_score}, and a Final Rating: {final_rating}, deliver a precise pedagogical assessment and concrete, actionable recommendations for module enhancement. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for module optimization.
"""

        user_perspective_assessment = ""
        instructor_feedback = ""

        try:
            # Generate user explanation
            user_response = self.gemini_model.generate_content(user_prompt_template)
            user_perspective_assessment = user_response.text.strip()
            print(f"Generated User Perspective Assessment: {user_perspective_assessment}")

            # Generate instructor explanation
            instructor_response = self.gemini_model.generate_content(instructor_prompt_template)
            instructor_feedback = instructor_response.text.strip()
            print(f"Generated Instructor Feedback: {instructor_feedback}")

            return {
                "User Perspective Assessment": user_perspective_assessment,
                "Instructor Feedback": instructor_feedback,
                "status": "success"
            }

        except Exception as e:
            print(f"Error generating final score explanations with Gemini: {e}")
            return {
                "User Perspective Assessment": "Fallback: Could not generate user-specific explanation for this module's final score.",
                "Instructor Feedback": "Fallback: Could not generate instructor-specific explanation for this module's final score.",
                "status": "fallback"
            }


    def generate_evaluation_explanation(self, course_metadata: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generates a concise evaluation explanation for both learner and instructor,
        displaying only the scores and the generated explanations per module.
        """
        course_title = course_metadata.get('Course', 'The Course')
        
        learner_explanation_parts = [
            f"Here's a summary of the course '{course_title}':\n"
        ]
        instructor_explanation_parts = [
            f"Detailed Evaluation Report for Course: '{course_title}'\n"
        ]

        for module_key, module_eval_data in evaluation_results.get('Module Evaluations', {}).items():
            # Get the module title, preferring 'Name' key, then 'Module Title', then fallback to module_key
            module_title = course_metadata.get(module_key, {}).get('Name', module_key)
            if module_title == module_key and isinstance(course_metadata.get(module_key), dict):
                module_title = course_metadata.get(module_key).get('Module Title', module_key)

            objective_syllabus_alignment_score = module_eval_data.get('objective_syllabus_alignment_score', {}).get('alignment_score', 0)
            course_syllabus_alignment_score = module_eval_data.get('course_syllabus_alignment_score', {}).get('alignment_score', 0)
            cognitive_complexity_score = module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('cognitive_complexity_score', 0)
            topic_coverage_score = module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('topic_coverage_score', 0)
            
            final_score_data = module_eval_data.get('final_score_and_explanation', {})
            final_score = final_score_data.get('final_score', 0)
            
            final_user_explanation = final_score_data.get('user_perspective_assessment', 'No user-specific explanation available for this module.')
            final_instructor_explanation = final_score_data.get('instructor_feedback', 'No instructor-specific explanation available for this module.')

            # Learner-facing module summary (concise)
            learner_explanation_parts.append(f"\n--- Module: {module_title} ---")
            learner_explanation_parts.append(f"Learning Objective-Syllabus Alignment Score: {round(objective_syllabus_alignment_score, 2)}/5")
            learner_explanation_parts.append(f"Course-Syllabus Alignment Score: {round(course_syllabus_alignment_score, 2)}/5")
            learner_explanation_parts.append(f"Cognitive Complexity Score: {round(cognitive_complexity_score, 2)}/5")
            learner_explanation_parts.append(f"Topic Coverage Score: {round(topic_coverage_score, 2)}/5")
            learner_explanation_parts.append(f"Overall Module Score: {round(final_score, 2)}/5")
            learner_explanation_parts.append(f"Explanation for you: {final_user_explanation}")
            
            # Instructor-facing module summary (concise)
            instructor_explanation_parts.append(f"\n--- Module: {module_title} ---")
            instructor_explanation_parts.append(f"Learning Objective-Syllabus Alignment Score: {round(objective_syllabus_alignment_score, 2)}/5")
            instructor_explanation_parts.append(f"Course-Syllabus Alignment Score: {round(course_syllabus_alignment_score, 2)}/5")
            instructor_explanation_parts.append(f"Cognitive Complexity Score: {round(cognitive_complexity_score, 2)}/5")
            instructor_explanation_parts.append(f"Topic Coverage Score: {round(topic_coverage_score, 2)}/5")
            instructor_explanation_parts.append(f"Overall Module Quality: {round(final_score, 2)}/5")
            instructor_explanation_parts.append(f"Explanation for instructor: {final_instructor_explanation}")
            instructor_explanation_parts.append("\n")

        learner_prompt = "\n".join(learner_explanation_parts)
        instructor_prompt = "\n".join(instructor_explanation_parts)

        return learner_prompt, instructor_prompt

    def get_gemini_final_score(self, objective_syllabus_alignment_score: float, course_syllabus_alignment_score: float, cognitive_complexity_score: float, topic_coverage_score: float) -> Dict[str, Any]:
        """
        Uses Gemini AI to synthesize alignment, cognitive complexity, and topic coverage scores into a final overall score
        for a given module. This function specifically gets the *score* from Gemini.
        """
        score_prompt = f"""
As an expert in educational course evaluation, synthesize the following scores for a module to provide a single overall module quality score (1-5).
This score should reflect the module's alignment, cognitive complexity, and topic coverage.

Scores:
- Learning Objective-Syllabus Alignment Score: {objective_syllabus_alignment_score}/5 (How well module content aligns with its objectives, addressing completeness, depth, relevance, and clarity of connection.)
- Course-Syllabus Alignment Score: {course_syllabus_alignment_score}/5 (How well module content aligns with overall course description, considering thematic cohesion, progression, scope appropriateness, and relevance to course outcomes.)
- Cognitive Complexity Score: {cognitive_complexity_score}/5 (Level of detail and complexity, including Bloom's Taxonomy levels and readability.)
- Topic Coverage Score: {topic_coverage_score}/5 (Range and diversity of topics covered.)

Provide:
1. Final Overall Score (1-5): A single float score representing the overall quality of this module, reflecting a holistic pedagogical assessment.

Respond in strict JSON:
{{
    "final_score": <float>
}}
"""
        final_score = 1.0 # Default fallback
        try:
            response = self.gemini_model.generate_content(score_prompt)
            response_text = response.text.strip()

            # Robust JSON extraction
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                response_text = response_text[start:end].strip()

            try:
                gemini_result = json.loads(response_text)
                final_score = round(float(gemini_result.get('final_score', 1.0)), 2)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error from Gemini (final score for module): {e}")
                print(f"Raw Gemini Response (attempting parse): {response_text}")
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        gemini_result = json.loads(json_match.group(0))
                        final_score = round(float(gemini_result.get('final_score', 1.0)), 2)
                        print("Successfully parsed JSON for final score from messy text.")
                    except json.JSONDecodeError:
                        print("Failed to parse JSON for final score even from regex match. Using fallback.")
                        pass # final_score remains 1.0
        except Exception as e:
            print(f"Error calling Gemini for final score for module: {e}")
            pass # final_score remains 1.0

        return {'final_score': final_score, 'status': 'success'}


    def get_gemini_final_score_and_explanation(self, objective_syllabus_alignment_score: float, course_syllabus_alignment_score: float, cognitive_complexity_score: float, topic_coverage_score: float) -> Dict[str, Any]:
        """
        Coordinates getting the final score and then its explanations.
        """
        # First, get the final score from Gemini
        final_score_result = self.get_gemini_final_score(objective_syllabus_alignment_score, course_syllabus_alignment_score, cognitive_complexity_score, topic_coverage_score)
        final_score = final_score_result['final_score']

        # Then, generate the specific user and instructor explanations using the new helper
        explanations = self._generate_final_score_explanations_with_gemini(objective_syllabus_alignment_score, course_syllabus_alignment_score, cognitive_complexity_score, topic_coverage_score, final_score)

        return {
            'final_score': final_score,
            'user_perspective_assessment': explanations['User Perspective Assessment'],
            'instructor_feedback': explanations['Instructor Feedback'],
            'status': 'success' if explanations['status'] == 'success' and final_score_result['status'] == 'success' else 'partial_success'
        }

    def _generate_course_level_explanations_with_gemini(self, course_title: str, module_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates course-level learner perspective assessment and instructor feedback
        based on *individual* module metrics.
        """
        module_data_strings = []
        for module_key, module_eval_data in module_evaluations.items():
            module_title = module_eval_data.get('module_name', module_key)
            obj_syllabus = module_eval_data.get('objective_syllabus_alignment_score', {}).get('alignment_score', 0)
            course_syllabus = module_eval_data.get('course_syllabus_alignment_score', {}).get('alignment_score', 0)
            cognitive_complexity = module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('cognitive_complexity_score', 0)
            topic_coverage = module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('topic_coverage_score', 0)
            final_module_score = module_eval_data.get('final_score_and_explanation', {}).get('final_score', 0)

            module_data_strings.append(f"""
    - Module: "{module_title}"
        - Objective-Syllabus Alignment: {obj_syllabus}/5
        - Course-Syllabus Alignment: {course_syllabus}/5
        - Cognitive Complexity: {cognitive_complexity}/5
        - Topic Coverage: {topic_coverage}/5
        - Final Module Score: {final_module_score}/5
            """)
        
        all_module_scores_detail = "\n".join(module_data_strings)

        # Learner Prompt for Course Level
        learner_course_prompt = f"""
Persona: You are simulating a learner's direct experience and perception of an *entire course's* quality. Your goal is to provide a personal reflection, as if you were the student who just completed the course.

As a learner, reflect on your overall course experience, focusing on how effectively the course's interconnected modules, learning goals, content, and intellectual demands came together to deliver a cohesive, accessible, and ultimately enriching learning journey.

**Individual Module Quality Observations (These influence my perceived experience across the course, but I should NOT state these values or their technical names explicitly in my reflection):**
{all_module_scores_detail}

Your response should adhere to the following guidelines:
* **Perspective:** From my personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** 3–5 sentences, describing my personal experience. Use phrases like "the course felt," "the content was," "my understanding grew."
* **Focus:** My perceived overall clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered across the entire course, synthesizing insights from individual modules.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names, module names, or numerical scores (e.g., "Module 1 scored high on alignment").

**Prompt Instruction:**
Given the detailed individual module quality observations for the entire course, write a concise learner-style reflection on the course's overall quality. Simulate how a learner experienced engaging with the course's content and structure across all modules. The last sentence should clearly imply the overall experience of the learner, considering potential variations between modules.
"""

        # Instructor Prompt for Course Level
        instructor_course_prompt = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on an *entire course's* design and its overall pedagogical effectiveness. Your insights aim to optimize the course for maximum learning impact and seamless integration.

As a pedagogical consultant, you are tasked with providing feedback on the overall course design to validate its effectiveness across all modules. This assessment aims to pinpoint overarching design strengths and identify strategic, actionable areas for improvement.

**Detailed Module Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
{all_module_scores_detail}

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor for the *entire course*.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 3–5 sentences, focusing on actionable recommendations at the course level. Avoid generic introductions.
* **Focus:** Identifying clear overall course design strengths, patterns across modules, and offering concrete, strategic improvements to the course's pedagogical efficacy, leveraging the underlying detailed module data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores for individual modules.

**Prompt Instruction:**
Given the detailed individual module quality observations for the entire course, deliver a precise pedagogical assessment and concrete, actionable recommendations for overall course enhancement. Synthesize all provided information, clearly articulating overarching areas for improvement or highlighting key strengths across the course's modules. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for holistic course optimization, considering performance variations between modules.
"""
        course_user_explanation = ""
        course_instructor_feedback = ""

        try:
            # Generate course-level user explanation
            user_response = self.gemini_model.generate_content(learner_course_prompt)
            course_user_explanation = user_response.text.strip()
            print(f"Generated Course-Level User Perspective Assessment: {course_user_explanation}")

            # Generate course-level instructor explanation
            instructor_response = self.gemini_model.generate_content(instructor_course_prompt)
            course_instructor_feedback = instructor_response.text.strip()
            print(f"Generated Course-Level Instructor Feedback: {course_instructor_feedback}")

            return {
                "course_user_perspective_assessment": course_user_explanation,
                "course_instructor_feedback": course_instructor_feedback,
                "status": "success"
            }
        except Exception as e:
            print(f"Error generating course-level explanations with Gemini: {e}")
            return {
                "course_user_perspective_assessment": "Fallback: Could not generate course-level user explanation.",
                "course_instructor_feedback": "Fallback: Could not generate course-level instructor feedback.",
                "status": "fallback"
            }


    def validate_course_from_drive(self, folder_id: str) -> Dict[str, Any]:
        """
        Validates a course by fetching metadata and applying various evaluation metrics module-wise.
        """
        course_metadata = self.fetch_metadata_json(folder_id)
        if not course_metadata:
            return {"status": "error", "message": "Failed to fetch course metadata."}

        results = {
            "course_title": course_metadata.get('Course', 'N/A'),
            "course_description": course_metadata.get('About this Course', 'No description provided.'),
            "Module Evaluations": {},
            "Course Level Evaluation": {} # New key for course-level evaluations
        }
        
        # Pass the entire course_metadata for relevance calculation to access all relevant fields
        course_data_for_alignment = {
            'About this Course': course_metadata.get('About this Course', 'No description provided.'),
            'Level': course_metadata.get('Level', 'Not specified.'),
            'Commitment': course_metadata.get('Commitment', 'Not specified.')
        }
        course_level = course_metadata.get('Level', 'Intermediate') # Default to Intermediate if not specified

        for key, value in course_metadata.items():
            if key.startswith('Module ') and isinstance(value, dict):
                module_key = key # This is "Module 1", "Module 2", etc.
                module_data = value
                
                # Determine the actual module title from metadata
                module_title = module_data.get('Name', module_data.get('Module Title', module_key))

                print(f"\n--- Processing {module_title} ---")

                module_eval_details = {}
                module_eval_details["module_name"] = module_title # Store the actual module name here

                # 1. Learning Objective-Syllabus Alignment Score (using Gemini)
                print(f"Calculating objective-syllabus alignment score for {module_title} using Gemini...")
                objective_syllabus_alignment_score_data = self.get_gemini_objective_syllabus_alignment_score(module_data)
                module_eval_details["objective_syllabus_alignment_score"] = objective_syllabus_alignment_score_data

                # 2. Course-Syllabus Alignment Score (using Gemini)
                print(f"Calculating course-syllabus alignment score for {module_title} using Gemini...")
                course_syllabus_alignment_score_data = self.get_gemini_course_syllabus_alignment_score(course_data_for_alignment, module_data)
                module_eval_details["course_syllabus_alignment_score"] = course_syllabus_alignment_score_data

                # 3. Bloom's Taxonomy Classification for Module Learning Objectives (using Gemini)
                print(f"Classifying Bloom's levels for {module_title}'s learning objectives using Gemini...")
                learning_objectives_bloom = []
                module_learning_objectives = module_data.get('Learning Objectives', [])
                for obj in module_learning_objectives:
                    bloom_data = self.get_gemini_bloom_classification(obj)
                    learning_objectives_bloom.append({
                        "objective": obj,
                        "bloom_classification": bloom_data
                    })
                module_eval_details["learning_objectives_bloom"] = learning_objectives_bloom

                # 4. Cognitive Complexity and Topic Coverage Score for the Module (using internal logic)
                print(f"Calculating cognitive complexity and topic coverage scores for {module_title}...")
                cognitive_complexity_topic_coverage_score_data = self.calculate_cognitive_complexity_and_topic_coverage(module_data, course_level, learning_objectives_bloom)
                module_eval_details["cognitive_complexity_and_topic_coverage_score"] = cognitive_complexity_topic_coverage_score_data


                # 5. Final Score and Explanation for the Module (using Gemini)
                print(f"Calculating final score and explanation for {module_title} using Gemini...")
                final_score_data = self.get_gemini_final_score_and_explanation(
                    objective_syllabus_alignment_score_data['alignment_score'],
                    course_syllabus_alignment_score_data['alignment_score'],
                    cognitive_complexity_topic_coverage_score_data['cognitive_complexity_score'],
                    cognitive_complexity_topic_coverage_score_data['topic_coverage_score']
                )
                module_eval_details["final_score_and_explanation"] = final_score_data

                results["Module Evaluations"][module_key] = module_eval_details

        # Generate course-level explanations using Gemini, passing ALL module evaluations (individual data)
        print("\nGenerating course-level explanations using Gemini...")
        course_level_explanations = self._generate_course_level_explanations_with_gemini(
            results["course_title"], results["Module Evaluations"] # Pass the full module evaluations dictionary
        )
        results["Course Level Evaluation"]["Course Explanations"] = course_level_explanations


        # Generate final explanation prompts for the whole course, incorporating module-wise insights
        learner_prompt, instructor_prompt = self.generate_evaluation_explanation(course_metadata, results)
        results["learner_explanation_prompt"] = learner_prompt
        results["instructor_explanation_prompt"] = instructor_prompt

        return results

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save the validation results to a JSON file, filtering for requested output."""
        
        filtered_results = {
            "Course Title": results.get("course_title"),
            "Module Evaluations": {},
            "Course Level Evaluation": {} # Include the new course-level evaluations
        }

        total_module_scores = 0
        module_count = 0

        for module_key, module_eval_data in results.get('Module Evaluations', {}).items():
            module_title_for_save = module_eval_data.get('module_name', module_key)

            validation_score = module_eval_data.get('final_score_and_explanation', {}).get('final_score')
            if validation_score is not None:
                total_module_scores += validation_score
                module_count += 1

            filtered_results["Module Evaluations"][module_key] = {
                "Module Name": module_title_for_save,
                "Objective Syllabus Alignment Score": module_eval_data.get('objective_syllabus_alignment_score', {}).get('alignment_score'),
                "Course Syllabus Alignment Score": module_eval_data.get('course_syllabus_alignment_score', {}).get('alignment_score'),
                "Cognitive Complexity Score": module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('cognitive_complexity_score'),
                "Topic Diversity Score": module_eval_data.get('cognitive_complexity_and_topic_coverage_score', {}).get('topic_coverage_score'),
                "Final Score And Explanation": {
                    "Validation Score": validation_score,
                    "Learner Perspective Assessment": module_eval_data.get('final_score_and_explanation', {}).get('user_perspective_assessment'),
                    "Instructor Feedback": module_eval_data.get('final_score_and_explanation', {}).get('instructor_feedback')
                }
            }
        
        # Calculate and add the average course validation score
        average_course_validation_score = round(total_module_scores / module_count, 2) if module_count > 0 else 0.0

        if "Course Level Evaluation" in results:
            filtered_results["Course Level Evaluation"]["Course Validation Score (Average of Modules)"] = average_course_validation_score
            filtered_results["Course Level Evaluation"]["Course Explanations"] = {
                "Course Learner Perspective Assessment": results["Course Level Evaluation"].get("Course Explanations", {}).get("course_user_perspective_assessment"),
                "Course Instructor Feedback": results["Course Level Evaluation"].get("Course Explanations", {}).get("course_instructor_feedback")
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
        print(f"Validation results saved to {filename}")


def main():
  
    SERVICE_ACCOUNT_FILE = r""
    gemini_api_key = "" # Replace with your actual Gemini API Key
    DRIVE_FOLDER_ID = "" # Replace with your Google Drive folder ID

    try:
        # Initialize Google Drive service outside the class
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)

        validator = SyllabusValidator(drive_service, gemini_api_key)
        results = validator.validate_course_from_drive(DRIVE_FOLDER_ID)
        validator.save_results(results, 'syl_val_report.json')
        print("\nValidation completed. Report saved.")
    except Exception as e:
        print(f"\nAn error occurred during validation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

if __name__ == "__main__":
    main()