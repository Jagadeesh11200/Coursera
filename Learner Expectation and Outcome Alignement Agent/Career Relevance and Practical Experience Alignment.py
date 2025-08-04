import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import io
import pandas as pd
import time
import os
import google.generativeai as genai
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer, util
import torch
import nltk

# Download NLTK punkt tokenizer data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except Exception as e:
    print(f"Required NLTK tokenizers not found, attempting download. Error: {e}")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        print("NLTK punkt and punkt_tab tokenizers downloaded and found.")
    except Exception as e_after_download:
        print(f"Failed to find required NLTK tokenizers even after download: {e_after_download}")
        exit()

# Initialize GPU acceleration
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("No GPU available, using CPU")

genai.configure(api_key = "")
t_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")

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

def get_cr_score(metadata):
    """
    Calculates the Career Relevance Score by using a combination of Gemini for
    synthesizing expectations and a sentence transformer for semantic similarity.
    """
    try:
        from nltk.tokenize import sent_tokenize

        # Step 1: Extract and synthesize user job expectations using Gemini
        about_text = metadata.get("About this Course", "")
        level = metadata.get("Level", "")
        prerequisites = metadata.get("Prerequisites", {})
        tech_prereqs = prerequisites.get("Technical Prerequisites", [])
        conceptual_prereqs = prerequisites.get("Conceptual Prerequisites", [])

        # Construct a prompt for Gemini to synthesize user expectations
        synthesis_prompt = f"""
        Analyze the following course information to synthesize a clear and concise list of the learner's implied professional and career expectations for this course.
        
        Course Description (About this Course): {about_text}
        Course Level: {level}
        Technical Prerequisites: {', '.join(tech_prereqs)}
        Conceptual Prerequisites: {', '.join(conceptual_prereqs)}
        
        Provide a bulleted list of 3-5 key expectations a learner would have regarding career outcomes after completing this course.
        """
        
        print("Using Gemini to synthesize user expectations...")
        response = model.generate_content(synthesis_prompt)
        gemini_expectations = [line.strip() for line in response.text.split('\n') if line.strip()]
        
        if not gemini_expectations:
            print("Gemini failed to synthesize expectations. Falling back to original method.")
            user_expectations = sent_tokenize(about_text)
            if level:
                user_expectations.append(f"The course is for {level} learners.")
            all_prereqs = [str(p) for p in (tech_prereqs + conceptual_prereqs) if p is not None]
            if all_prereqs:
                user_expectations.append(f"Learners should have the following prerequisites: {', '.join(all_prereqs)}.")
        else:
            user_expectations = gemini_expectations
            
        print(f"Synthesized user expectations: {user_expectations}")

        # Step 2: Extract learning objectives
        learning_objectives = []
        for key, value in metadata.items():
            if key.startswith("Module") and isinstance(value, dict):
                module_los = value.get("Learning Objectives", [])
                if isinstance(module_los, list):
                    learning_objectives.extend([str(lo) for lo in module_los if lo is not None])
                elif module_los is not None:
                    learning_objectives.append(str(module_los))

        if not user_expectations or not learning_objectives:
            print("Missing user expectations or learning objectives. CR Score cannot be calculated.")
            return 1.0, 0.0

        # Step 3: Semantic comparison (embedding similarity)
        expectation_alignment = []
        for expectation in user_expectations:
            try:
                exp_embedding = t_model.encode(expectation, convert_to_tensor=True, device=device)
                lo_embeddings = t_model.encode(learning_objectives, convert_to_tensor=True, device=device)
                
                best_match_score = util.cos_sim(exp_embedding, lo_embeddings).max().item()
                expectation_alignment.append(best_match_score)
            except Exception as e_encode:
                print(f"Error encoding or comparing expectation '{expectation}': {e_encode}")

        # Step 4: Generate alignment report and calculate scores
        valid_scores = [score for score in expectation_alignment if score is not None]
        if valid_scores:
            avg_similarity = sum(valid_scores) / len(valid_scores)
            intermediate_embedding_score = avg_similarity
            cr_score_1_5 = round(avg_similarity * 4 + 1, 2)
        else:
            print("No valid similarity scores to compute average. Setting CR score to 1.0 (lowest).")
            cr_score_1_5 = 1.0
            intermediate_embedding_score = 0.0
        
        return cr_score_1_5, intermediate_embedding_score

    except Exception as e:
        print(f"[ERROR] Career Relevance scoring failed: {e}")
        return 1.0, 0.0

def get_pea_score(metadata):
    """
    Calculates the Practical Experience Alignment Score using Gemini to identify
    implicit promises and a semantic model to match them to activities.
    """
    # 1. Use Gemini to extract both explicit and implicit hands-on promises
    about_text = metadata.get("About this Course", "")
    prerequisites = metadata.get("Prerequisites", {})
    tech_prereqs = prerequisites.get("Technical Prerequisites", [])
    conceptual_prereqs = prerequisites.get("Conceptual Prerequisites", [])
    learning_objectives_text = []
    for key, value in metadata.items():
        if key.startswith("Module") and isinstance(value, dict):
            los = value.get("Learning Objectives", [])
            learning_objectives_text.extend([str(lo) for lo in los if isinstance(lo, str)])

    synthesis_prompt = f"""
    Analyze the following course information to identify all promises of practical, hands-on experience for the learner. Include both explicit statements (e.g., "build a project") and implicit promises (e.g., "master machine learning models" implies a project).
    
    Course Description: {about_text}
    Prerequisites: {', '.join(tech_prereqs + conceptual_prereqs)}
    Learning Objectives: {'; '.join(learning_objectives_text)}
    
    Provide a bulleted list of all 3-5 key practical promises.
    """
    
    print("\nUsing Gemini to synthesize hands-on promises...")
    response = model.generate_content(synthesis_prompt)
    hands_on_promises = [line.strip() for line in response.text.split('\n') if line.strip()]
    
    if not hands_on_promises:
        print("Gemini failed to synthesize hands-on promises. PEA Score cannot be calculated.")
        return 1.0, 0.0

    print(f"\nSynthesized practical promises: {hands_on_promises}")

    # 2. Extract practical activities from syllabus
    practical_texts = []
    for key in metadata:
        if key.startswith("Module") and isinstance(metadata[key], dict):
            syllabus_items = metadata[key].get("Syllabus", [])
            if isinstance(syllabus_items, list):
                for item in syllabus_items:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        if any(k in item_lower for k in ["quiz", "test", "exam", "lab", "project", "case study", "assignment", "exercise", "graded"]) and "video" not in item_lower:
                            practical_texts.append(item.strip())

    if not practical_texts:
        print("No practical activities found in syllabus. PEA Score cannot be calculated.")
        return 1.0, 0.0

    print(f"Practical activities extracted: {practical_texts}")

    # 3. Encode texts and compute cosine similarity
    try:
        hands_on_embeddings = t_model.encode(hands_on_promises, convert_to_tensor=True, device=device)
        practical_embeddings = t_model.encode(practical_texts, convert_to_tensor=True, device=device)
        cosine_scores = util.cos_sim(hands_on_embeddings, practical_embeddings)
        max_similarities = cosine_scores.max(dim=1).values.cpu().tolist()
    except Exception as e:
        print(f"Error during embedding generation for PEA score: {e}")
        return 1.0, 0.0

    # 4. Generate alignment report and calculate scores
    pea_alignment_report = []
    for i, expectation in enumerate(hands_on_promises):
        best_match_index = cosine_scores[i].argmax().item()
        best_matched_activity = practical_texts[best_match_index]
        similarity_score = round(max_similarities[i], 3)
        pea_alignment_report.append(f"- Promise: {expectation}\n   Matched Activity: {best_matched_activity}\n   Similarity: {similarity_score}")

    print("\n--- Practical Experience Alignment Report ---")
    print("\n".join(pea_alignment_report))

    if max_similarities:
        avg_similarity = sum(max_similarities) / len(max_similarities)
        intermediate_embedding_score = avg_similarity
        pea_score_0_1 = round(avg_similarity, 3)
        pea_score_1_5 = round(pea_score_0_1 * 4 + 1, 2)
    else:
        avg_similarity = 0.0
        pea_score_1_5 = 1.0
        intermediate_embedding_score = 0.0

    return pea_score_1_5, intermediate_embedding_score

def prompt_career_relevance_alignment(intermediate_embedding_score_cr, final_score):
    """
    Generates tailored prompts for the "Career Relevance Alignment" metric.
    """
    user_prompt_template_cr = f"""
Persona: You are simulating a learner's direct experience and perception of a course's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the course.

As a learner, reflect on your course experience, focusing on how effectively the course's content and learning outcomes align with your professional goals and career expectations.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**

* **Career Relevance (CR) (Score Range: 1-5):**
    * **Definition:** This metric indicates how well the course's learning objectives and content align with explicit or implied career expectations, as outlined in the course description and prerequisites.
    * **Scoring:** 1 (significant disconnect between content and career goals) to 5 (perfect congruence and direct application to career path).
    
* **Underlying Semantic Similarity Score (Score Range: 0-1):**
    * **Definition:** This metric is an internal measure of how semantically similar the course's content is to the professional outcomes and job expectations described in the course materials. A higher score indicates a stronger conceptual link.
    * **Scoring:** 0 (no semantic similarity) to 1 (perfect semantic similarity).

**Underlying Quality Observations (These influence the perceived experience, but should NOT be stated in the reflection):**
- Career Relevance Score: {final_score}/5
- Underlying Semantic Similarity Score: {intermediate_embedding_score_cr}

The response should adhere to the following guidelines:
* **Perspective:** From a personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, concise and objective.
* **Style:** 2–4 sentences, describing the learner's experience. Use passive voice or objective statements; avoid any first-person pronouns (I, my, me).
* **Focus:** Perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on career relevance").

**Prompt Instruction:**
Given the underlying quality observations: Career Relevance Score: {final_score}, and the Underlying Semantic Similarity Score: {intermediate_embedding_score_cr}, write a short learner-style reflection on the course's quality. Simulate a learner's experience engaging with the course's content and structure. The last sentence should clearly imply the overall experience on career relevance.
"""

    # Instructor Feedback Prompt
    instructor_prompt_template_cr = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a course's design and its overall pedagogical effectiveness. Your insights aim to optimize the course for maximum learning impact and seamless integration within the broader curriculum.

As a pedagogical consultant, you are tasked with providing feedback on a course's syllabus to validate its design and effectiveness. This assessment aims to pinpoint design strengths and identify actionable areas for improvement.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Career Relevance (CR) (Score Range: 1-5):**
    * **Definition:** This metric quantifies how precisely the course's stated learning objectives and content directly correspond to the professional outcomes or job expectations stated in the course's promotional materials or prerequisites.
    * **Scoring:** 1 (significant disconnects or content deficiencies relative to career expectations) to 5 (perfect congruence; all objectives are fully supported, clearly addressed, and content is optimally aligned with stated career goals).
    
* **Underlying Semantic Similarity Score (Score Range: 0-1):**
    * **Definition:** This metric is an internal measure of how semantically similar the course's content is to the professional outcomes and job expectations described in the course materials. It is a foundational metric for the final Career Relevance Score.
    * **Scoring:** 0 (no semantic similarity) to 1 (perfect semantic similarity).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Career Relevance Score: {final_score}/5
- Underlying Semantic Similarity Score: {intermediate_embedding_score_cr}

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Career Relevance Score: {final_score}, and the Underlying Semantic Similarity Score: {intermediate_embedding_score_cr}, deliver a precise pedagogical assessment and concrete, actionable recommendations for course enhancement. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
    return user_prompt_template_cr, instructor_prompt_template_cr

def prompt_practical_experience_alignment(intermediate_embedding_score_pea, final_score):
    """
    Generates tailored prompts for the "Practical Experience Alignment" metric.
    """

    user_prompt_template_pea = f"""
Persona: You are simulating a learner's direct experience and perception of a course's quality. Your goal is to provide a personal reflection, as if you were the student who just completed the course.

As a learner, reflect on your course experience, focusing on how effectively the course's content has prepared you for professional and ethical challenges in your field.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**

* **Professional and Ethical Alignment (PEA) (Score Range: 1-5):**
    * **Definition:** This metric indicates how well the course's content and activities prepare learners to navigate the professional and ethical landscape of their target field.
    * **Scoring:** 1 (course content is lacking in professional/ethical context) to 5 (course provides comprehensive and practical guidance on professional and ethical conduct).
    
* **Underlying Semantic Similarity Score (Score Range: 0-1):**
    * **Definition:** This metric is an internal measure of how semantically similar the course's content is to the professional outcomes and job expectations described in the course materials. A higher score indicates a stronger conceptual link.
    * **Scoring:** 0 (no semantic similarity) to 1 (perfect semantic similarity).

**Underlying Quality Observations (These influence the perceived experience, but should NOT be stated in the reflection):**
- Professional and Ethical Alignment Score: {final_score}/5
- Underlying Semantic Similarity Score: {intermediate_embedding_score_pea}

The response should adhere to the following guidelines:
* **Perspective:** From a personal, simulated point of view as the learner.
* **Tone:** Reflective, authentic, concise and objective.
* **Style:** 2–4 sentences, describing the learner's experience. Use passive voice or objective statements; avoid any first-person pronouns (I, my, me).
* **Focus:** Perceived clarity, intellectual engagement, relevance of content, and how comprehensively topics were covered, all based on the underlying data.
* **Avoid:** Technical jargon, generic introductions, or directly mentioning the metric names or numerical scores (e.g., "the course scored high on professional alignment").

**Prompt Instruction:**
Given the underlying quality observations: Professional and Ethical Alignment Score: {final_score}, and the Underlying Semantic Similarity Score: {intermediate_embedding_score_pea}, write a short learner-style reflection on the course's quality. Simulate a learner's experience engaging with the course's content and structure. The last sentence should clearly imply the overall experience for practical experience alignment.
"""

    instructor_prompt_template_pea = f"""
You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on a course's design and its overall pedagogical effectiveness. Your insights aim to optimize the course for maximum learning impact and seamless integration within the broader curriculum.

As a pedagogical consultant, you are tasked with providing feedback on a course's syllabus to validate its design and effectiveness. This assessment aims to pinpoint design strengths and identify actionable areas for improvement.

**Evaluation Metrics & Scoring Principles (for your understanding):**

* **Professional and Ethical Alignment (PEA) (Score Range: 1-5):**
    * **Definition:** This metric rigorously assesses how coherently and strategically this specific course's content addresses professional and ethical standards, practices, and challenges relevant to the field of study. It confirms the course's meaningful contribution to the larger curriculum's professional development goals.
    * **Scoring:** 1 (course content is largely disconnected from professional/ethical standards) to 5 (course content seamlessly integrates, significantly contributes to, and is perfectly scoped within the overall professional and ethical framework of the field).
    
* **Underlying Semantic Similarity Score (Score Range: 0-1):**
    * **Definition:** This metric is an internal measure of how semantically similar the course's content is to the professional outcomes and job expectations described in the course materials. A higher score indicates a stronger conceptual link.
    * **Scoring:** 0 (no semantic similarity) to 1 (perfect semantic similarity).

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
- Professional and Ethical Alignment Score: {final_score}/5
- Underlying Semantic Similarity Score: {intermediate_embedding_score_pea}

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise and concise assessment within 2–4 sentences, focusing on actionable recommendations. Avoid generic introductions.
* **Focus:** Identifying clear design strengths and offering concrete, strategic improvements to the course's pedagogical efficacy, leveraging the underlying data.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores.

**Prompt Instruction:**
Given the underlying quality observations: Professional and Ethical Alignment Score: {final_score}, and the Underlying Semantic Similarity Score: {intermediate_embedding_score_pea}, deliver a precise pedagogical assessment and concrete, actionable recommendations for course enhancement. Synthesize all provided information, clearly articulating areas for improvement or highlighting key strengths. The final sentence must be a concise summary of the overall feedback, unequivocally outlining a primary action item for course optimization.
"""
    return user_prompt_template_pea, instructor_prompt_template_pea

def learner_expectation_course_outcome_analysis(service_account_file, folder_id):
    service = get_gdrive_service(service_account_file)
    metadata = fetch_metadata_json(service, folder_id)

    if metadata is None:
        print("Failed to retrieve metadata.json. Cannot perform analysis.")
        return {"error": "Failed to retrieve metadata."}

    output = {}
    output["Course Level Evaluation"] = {}

    print("\n--- Calculating Career Relevance Score ---")
    cr_score, cr_intermediate_embedding_score = get_cr_score(metadata)
    
    cr_user_prompt, cr_instructor_prompt = prompt_career_relevance_alignment(
        intermediate_embedding_score_cr=cr_intermediate_embedding_score,
        final_score=cr_score 
    )

    cr_user_assessment = model.generate_content(cr_user_prompt).text.strip()
    cr_instructor_assessment = model.generate_content(cr_instructor_prompt).text.strip()

    output["Course Level Evaluation"]["Career Relevance"] = {
        "Career Relevance Score": cr_score, 
        "Learner Perspective Assessment": cr_user_assessment,
        "Instructor Feedback": cr_instructor_assessment
    }

    print("\n--- Calculating Practical Experience Alignment Score ---")
    pea_score, pea_intermediate_embedding_score = get_pea_score(metadata)
    
    pea_user_prompt, pea_instructor_prompt = prompt_practical_experience_alignment(
        intermediate_embedding_score_pea=pea_intermediate_embedding_score,
        final_score=pea_score
    )

    pea_user_assessment = model.generate_content(pea_user_prompt).text.strip()
    pea_instructor_assessment = model.generate_content(pea_instructor_prompt).text.strip()

    output["Course Level Evaluation"]["Practical Experience Alignment"] = {
        "Practical Experience Alignment Score": pea_score, 
        "Learner Perspective Assessment": pea_user_assessment,
        "Instructor Feedback": pea_instructor_assessment
    }

    return output

if __name__ == "__main__":
    service_account_file = r""
    folder_id = ""
    start_time = time.time()
    analysis_results = learner_expectation_course_outcome_analysis(service_account_file, folder_id)
    print("\n--- Analysis Results ---")
    print(json.dumps(analysis_results, indent=4))
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTime taken: {elapsed_minutes:.2f} minutes")

    output_filename = "Learner_expectation_output_sample.json"
    with open(output_filename, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    print(f"\nAnalysis results saved to {output_filename}")
