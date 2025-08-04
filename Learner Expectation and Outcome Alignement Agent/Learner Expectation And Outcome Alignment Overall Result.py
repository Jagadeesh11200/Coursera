import json
import re
import os
import google.generativeai as genai

def extract_course_level_data(json_content):
    """
    Extracts scores and feedback from the 'Course Level Evaluation' section of a JSON.
    This function is now designed to be called with the *content* of the 'Course Level Evaluation'
    dictionary, regardless of its nesting level. It handles both multi-score and single-score structures
    within that dictionary.

    Returns a list of dictionaries, each containing 'name', 'score', 'learner_feedback',
    and 'instructor_feedback'.
    """
    extracted_entries = []

    # Check for the multi-score structure by seeing if any value is a dictionary
    if any(isinstance(v, dict) for v in json_content.values()):
        for key, value_dict in json_content.items():
            if isinstance(value_dict, dict):
                score = "N/A"
                learner_fb_text = ""
                instructor_fb_text = ""

                # Iterate through the inner dictionary to find score and feedback
                for sub_key, sub_value in value_dict.items():
                    # Find the score key (e.g., "Career Relevance Score")
                    if "score" in sub_key.lower():
                        score = sub_value
                    # Find the learner feedback key
                    elif "learner perspective assessment" in sub_key.lower():
                        learner_fb_text = sub_value
                    # Find the instructor feedback key
                    elif "instructor feedback" in sub_key.lower():
                        instructor_fb_text = sub_value

                # Create a clean name for the metric
                metric_name = key.replace('_', ' ').title()

                extracted_entries.append({
                    "name": metric_name,
                    "score": score,
                    "learner_feedback": learner_fb_text,
                    "instructor_feedback": instructor_fb_text
                })
    else:
        # Handle single-score structure
        # Look for any key that contains the word 'score'
        score = "N/A"
        metric_name = "N/A"
        learner_fb = json_content.get("Course Learner Perspective Assessment", "")
        instructor_fb = json_content.get("Course Instructor Feedback", "")

        for key, value in json_content.items():
            if "score" in key.lower():
                score = value
                # Create a clean name from the score key
                metric_name = key.replace('Score', '').strip()

        if metric_name != "N/A" or score != "N/A":
            extracted_entries.append({
                "name": metric_name,
                "score": score,
                "learner_feedback": learner_fb,
                "instructor_feedback": instructor_fb
            })

    return extracted_entries


def find_and_process_course_eval(obj, all_extracted_data):
    """
    Recursively searches for 'Course Level Evaluation' key in a dictionary or list
    and processes its content.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "Course Level Evaluation" and isinstance(value, dict):
                extracted_data = extract_course_level_data(value)
                all_extracted_data.extend(extracted_data)
            else:
                find_and_process_course_eval(value, all_extracted_data)
    elif isinstance(obj, list):
        for item in obj:
            find_and_process_course_eval(item, all_extracted_data)

def generate_overall_rating_prompt(scores_data):
    """
    Generates a highly detailed and specific prompt for the Academic Oversight Committee persona.
    Includes dynamic scores and a comprehensive, hardcoded methodology, while explicitly
    mentioning the scores in the prompt instruction and eliminating any request for justification.
    """
    scores_list_str = "\n".join([f"- {item['name']} Score: {item['score']:.2f}/5.0" for item in scores_data if isinstance(item['score'], (int, float))])

    if not scores_list_str:
        return "No valid scores were provided for overall rating calculation."

    return f"""
Persona: You are a distinguished member of an Academic Oversight Committee, a panel known for its rigorous and impartial analysis. Your role is to perform a data-driven assessment of a course's overall quality. Your reputation is built on your ability to synthesize diverse validation metrics into a single, authoritative, and defensible rating.

Your primary objective is to meticulously calculate an overall course quality rating on a scale of 1.0 to 5.0. This rating must be a direct and precise reflection of the comprehensive quality indicated by several key validation scores, each representing a crucial dimension of the course's design and content. The final output must be a single floating-point number, rounded to exactly two decimal places, representing your final, authoritative assessment.

**Evaluation Methodology (Guiding Principles for Qualitative Synthesis):**

* **Career Relevance (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses the direct applicability of the course content to professional skills and industry-specific requirements. It measures how well the course prepares a learner for the demands of the modern workforce.
    * **Scoring Context:** A high score (e.g., 4.5+) indicates a strong, explicit link between the course curriculum and career readiness, with content directly aligning with in-demand professional competencies. A low score (e.g., below 2.5) suggests the course is theoretical or abstract, lacking practical application or direct career benefits.

* **Practical Experience Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This measures the effectiveness of the course in integrating theoretical concepts with real-world professional scenarios, case studies, and ethical dilemmas. It evaluates the bridge between abstract knowledge and tangible application.
    * **Scoring Context:** A high score signifies a curriculum rich with hands-on projects, simulations, and problem-based learning. A lower score suggests a significant gap between theory and practical application, with content remaining largely conceptual.

* **Time Value Balance (Score Range: 1.0-5.0):**
    * **Definition:** This evaluates the efficiency and pedagogical value of the course relative to the time investment required from the learner. It's a measure of content density versus learning efficacy.
    * **Scoring Context:** A high score indicates a well-aligned and efficient learning experience where the time invested yields a proportional and significant educational return. A low score may point to content density issues, poor pacing, or redundant information that detracts from the value-per-hour.

* **Cognitive Depth Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This analyzes the course's design for intellectual rigor and its progression through different levels of Bloom's Taxonomy, ensuring it challenges learners beyond simple recall.
    * **Scoring Context:** A high score reflects a strong emphasis on higher-order thinking tasks, such as analysis, synthesis, and evaluation. It means the course effectively pushes learners to apply, analyze, and critique information. A low score indicates the course relies too heavily on foundational knowledge and memorization without fostering deeper intellectual engagement.

**Underlying Quality Observations (Precise numerical inputs for your assessment):**
{scores_list_str}

**Overall Rating Determination (Your Mandate):**
-   The final overall course quality rating will be determined by a **holistic assessment** of all individual scores provided. This is not a simple arithmetic calculation (e.g., an average or weighted average), but a reasoned, qualitative judgment based on the comprehensive strengths and weaknesses indicated by each dimension.
-   You must evaluate the collective impact of the scores. A low score in one critical area (e.g., Practical Experience Alignment) could significantly downgrade the overall rating, even if other scores are high. Conversely, consistently strong scores across all dimensions should lead to an excellent overall rating.
-   The final overall rating must be presented as a single floating-point number, rounded to exactly two decimal places, in the required output format.

**Expected Response Guidelines:**
* **Perspective:** Assume the authoritative, analytical point of view of a member of the Academic Oversight Committee.
* **Tone:** Formal, impartial, and highly data-driven.
* **Style:** Concise and direct. Avoid conversational fillers, subjective opinions, or overly descriptive language.
* **Focus:** Directly provide the calculated overall rating. Do not include any explanations, justifications, or reasoning.
* **Avoid:** Any mention of specific mathematical formulas or the inclusion of qualitative feedback from learners or instructors.

**Prompt Instruction:**
Given the underlying quality observations of all provided scores, specifically **Career Relevance**, **Practical Experience Alignment**, **Time Value Balance**, and **Cognitive Depth Alignment**, determine the single overall course quality rating out of 5.0 through a holistic assessment, adhering strictly to the persona and output format.

**Required Output Format:**
Overall Course Rating: [Your Assessed Rating out of 5.0 with two decimal points]
"""

def generate_learner_perspective_summary_prompt(feedback_data):
    """
    Generates a detailed prompt for an AI model to summarize learner feedback.
    Includes a hardcoded methodology and mandated areas string.
    """
    mandated_areas_str = """
1. Career Relevance and its connection to the learner experience
2. Practical Experience Alignment and its connection to the learner experience
3. Time Value Balance and its connection to the learner experience
4. Cognitive Depth Alignment and its connection to the learner experience
"""
    feedback_observations_str = "\n".join([
        f"- From {item['name']} Learner Feedback (Score: {item['score']:.2f}/5.0): \"{item['learner_feedback'].strip()}\""
        for item in feedback_data if item['learner_feedback'].strip() and isinstance(item['score'], (int, float))
    ])

    return f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality and cohesiveness. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's various interconnected elements came together to deliver a cohesive, accessible, and ultimately enriching learning experience. Your output should directly convey your feelings and observations about the course's effectiveness.

**Evaluation Methodology (referencing each metric's specific mechanism):**
These categories represent the foundational metrics used to inform the overall assessment. The final evaluation is implicitly shaped by these distinct sources of feedback, each contributing to a comprehensive understanding of the course's quality.
* **Career Relevance (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses the direct applicability of course content to professional skills and industry expectations.
    * **Scoring Context:** A high score indicates a strong, explicit link between the course and career readiness.
* **Practical Experience Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This measures the integration of theoretical concepts with real-world professional scenarios, case studies, and ethical dilemmas.
    * **Scoring Context:** A lower score suggests a gap between theory and practical application.
* **Time Value Balance (Score Range: 1.0-5.0):**
    * **Definition:** This evaluates the efficiency of the course in delivering educational value relative to the time invested.
    * **Scoring Context:** A lower score may indicate content density or pacing issues, while a higher score signifies a well-aligned and efficient learning experience.
* **Cognitive Depth Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This analyzes the course's design for intellectual rigor and its progression through different levels of Bloom's taxonomy.
    * **Scoring Context:** A high score reflects a strong emphasis on higher-order thinking tasks like analysis and evaluation.

**Underlying Quality Observations (These are the direct textual observations influencing my perceived experience, along with their associated quality scores. Use these scores to gauge the *severity* or *strength* of the feedback, but DO NOT explicitly state their technical names, sources, or numerical values in my reflection):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 3 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated reflection **MUST be a maximum of 3 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly convey observations related to the learner's experience regarding:**

{mandated_areas_str}

Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using connecting phrases to maintain flow. Each sentence should contribute meaningfully to the overall assessment, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** A highly cohesive and evaluative narrative of **2–3 sentences ONLY**. This narrative should build towards an overarching impression of the learning experience, strategically incorporating observations for each mandated analytical area without sounding like a list. **Start directly with the content, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list feedback points.** Instead, **holistically analyze and integrate** all underlying feedback to form a unified assessment of the course's overall effectiveness, clarity, and engagement. The narrative should seamlessly combine observations about how the various course elements, interpreted through the context of their specific methodologies, collectively shaped the experience, subtly emphasizing strengths or weaknesses based on associated scores. For instance, a low score accompanying a feedback point should imply a more significant perceived deficiency that contributes to the overall feeling.
* **Avoid:** Technical jargon, generic introductions, directly mentioning the metric names or numerical scores, or explicit reference to the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various learner feedback sources, including their associated quality scores, **conduct a thorough analysis to form a unified learner-style reflection** on the overall quality and cohesiveness of the course. **Integrate all insights** into a coherent narrative that simulates how the learner *perceived* the course's structure, content, and clarity, referencing the specific methodologies behind each evaluated aspect. **YOUR 2-3 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE LEARNER'S EXPERIENCE REGARDING THE CAREER RELEVANCE, PRACTICAL EXPERIENCE, COGNITIVE DEPTH, AND TIME VALUE.** The final sentence must be a concise, overarching assessment of the course's effectiveness from a learner's vantage point, acting as a clear key takeaway.

**Required Output Format:**
[A holistic summary of the course's effectiveness from the learner's point of view, reflecting on how well the content connected to career goals, the balance between practical application and theory, the overall value for the time spent, and the depth of intellectual challenge. This should be a cohesive narrative of 2-3 sentences, without using first-person pronouns, and should start directly with the content.]
"""

def generate_instructor_perspective_summary_prompt(feedback_data):
    """
    Generates a prompt for an AI model to summarize instructor feedback.
    Includes a hardcoded methodology and a hardcoded mandated areas string.
    """
    mandated_areas_str = """
1. Career Relevance and its connection to the learner experience
2. Practical Experience Alignment and its connection to the learner experience
3. Time Value Balance and its connection to the learner experience
4. Cognitive Depth Alignment and its connection to the learner experience
"""
    feedback_observations_str = "\n".join([
        f"- From {item['name']} Instructor Feedback (Score: {item['score']:.2f}/5.0): \"{item['instructor_feedback'].strip()}\""
        for item in feedback_data if item['instructor_feedback'].strip() and isinstance(item['score'], (int, float))
    ])

    return f"""
Persona: You are a Lead Instructional Designer providing a comprehensive, objective analysis of a course's design and delivery based on aggregated instructor and system feedback. Your goal is to pinpoint specific strengths and weaknesses in the curriculum's structure, pedagogical approach, and content quality.

This report serves as a diagnostic tool for course improvement, highlighting areas where the design excels and where revisions are most critical. Your analysis should be grounded in the provided feedback, focusing on actionable insights for enhancing the learning experience.

**Evaluation Methodology (referencing each metric's specific mechanism):**
These categories represent the foundational metrics used to inform the overall assessment. The final evaluation is implicitly shaped by these distinct sources of feedback, each contributing to a comprehensive understanding of the course's quality.
* **Career Relevance (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses the direct applicability of course content to professional skills and industry expectations.
    * **Scoring Context:** A high score indicates a strong, explicit link between the course and career readiness.
* **Practical Experience Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This measures the integration of theoretical concepts with real-world professional scenarios, case studies, and ethical dilemmas.
    * **Scoring Context:** A lower score suggests a gap between theory and practical application.
* **Time Value Balance (Score Range: 1.0-5.0):**
    * **Definition:** This evaluates the efficiency of the course in delivering educational value relative to the time invested.
    * **Scoring Context:** A lower score may indicate content density or pacing issues, while a higher score signifies a well-aligned and efficient learning experience.
* **Cognitive Depth Alignment (Score Range: 1.0-5.0):**
    * **Definition:** This analyzes the course's design for intellectual rigor and its progression through different levels of Bloom's taxonomy.
    * **Scoring Context:** A high score reflects a strong emphasis on higher-order thinking tasks like analysis and evaluation.

**Underlying Quality Observations (These are the direct textual observations influencing my professional judgment, along with their associated quality scores. Use these scores to gauge the *severity* or *strength* of the feedback, but DO NOT explicitly state their technical names, sources, or numerical values in my report):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 3 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated report **MUST be a maximum of 3 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly convey professional observations related to:**
{mandated_areas_str}
Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using connecting phrases to maintain a professional flow. Each sentence should contribute meaningfully to the overall assessment, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the professional, analytical, and objective point of view of a Lead Instructional Designer.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** A highly cohesive and evaluative narrative of **2–3 sentences ONLY**. This narrative should build towards an overarching professional assessment of the course's design, strategically incorporating observations for each mandated analytical area without sounding like a list. **Start directly with the content, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list feedback points.** Instead, **holistically analyze and integrate** all underlying feedback to form a unified assessment of the course's design and delivery. The narrative should seamlessly combine observations about how the various course design elements, interpreted through the context of their specific methodologies, collectively shaped the instructional effectiveness, subtly emphasizing strengths or weaknesses based on associated scores. For instance, a low score accompanying a feedback point should imply a more significant design deficiency that requires immediate attention.
* **Avoid:** Technical jargon, generic introductions, directly mentioning the metric names or numerical scores, or explicit reference to the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various instructor feedback sources, including their associated quality scores, **conduct a thorough analysis to form a unified instructional designer-style report** on the course's design and delivery. **Integrate all insights** into a coherent narrative that simulates a professional's diagnostic assessment of the course's structure, pedagogical approach, and content quality, referencing the specific methodologies behind each evaluated aspect. **YOUR 2-3 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE INSTRUCTOR'S EXPERIENCE REGARDING THE CAREER RELEVANCE, PRACTICAL EXPERIENCE, COGNITIVE DEPTH, AND TIME VALUE.** The final sentence must be a concise, overarching assessment of the course's instructional effectiveness, acting as a clear key takeaway for future revisions.

**Required Output Format:**
[Your deeply analyzed and integrated summary of all instructor feedback, in an instructional designer's voice, **STRICTLY 2-3 sentences**, focusing on an overall professional assessment rather than just synthesis, and **analyzing each analytical area very clearly** including the curriculum's alignment with career relevance, its integration of practical experience, its design for cognitive depth, and its time value. **Start directly with the content.**]
"""

# --- MODIFIED: get_gemini_response with explicit API key and temperature ---
def get_gemini_response(prompt_text):
    """
    Sends a prompt to the Gemini API and returns the text response.
    Uses hardcoded API key and sets temperature=0 for deterministic behavior.
    """
    try:
        # CONFIGURE GENAI WITH YOUR HARDCODED API KEY (FOR DEMO/TESTING ONLY)
        genai.configure(api_key="")

        # Initialize Gemini 2.0 Flash (001) with temperature=0 for deterministic behavior
        model = genai.GenerativeModel("models/gemini-2.0-flash-001", generation_config={"temperature": 0.0})

        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        print("Please check your API key and network connection.")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # The file paths list is updated based on your previous and current instructions.
    json_file_paths = [
    ]

    all_extracted_data = []

    for file_path in json_file_paths:
        try:
            with open(file_path, 'r') as f:
                json_content = json.load(f)
            
            find_and_process_course_eval(json_content, all_extracted_data)
            print(f"Successfully processed data from '{file_path}'.")

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Skipping.")
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from '{file_path}'. {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_path}': {e}")
    
    # --- ADDED: Displaying all scores being considered ---
    print("\n" + "="*80)
    print("SCORES CONSIDERED FOR ANALYSIS:")
    print("="*80)
    if all_extracted_data:
        for item in all_extracted_data:
            print(f"Metric: {item['name']}, Score: {item['score']}")
    else:
        print("No scores were extracted from the provided files.")
    print("="*80)

    # --- Dynamically Build Score/Feedback Data from aggregated data ---
    learner_feedback_data = []
    instructor_feedback_data = []

    # Create feedback data lists from the extracted entries
    for entry in all_extracted_data:
        if isinstance(entry['score'], (int, float)):
            learner_feedback_data.append({
                "name": entry['name'],
                "score": entry['score'],
                "learner_feedback": entry['learner_feedback']
            })
            instructor_feedback_data.append({
                "name": entry['name'],
                "score": entry['score'],
                "instructor_feedback": entry['instructor_feedback']
            })

    # --- Generate Prompts with Dynamic Data ---
    overall_rating_prompt = generate_overall_rating_prompt(all_extracted_data)
    learner_prompt = generate_learner_perspective_summary_prompt(learner_feedback_data)
    instructor_prompt = generate_instructor_perspective_summary_prompt(instructor_feedback_data)

    # --- Display Generated Prompts ---
    print("\n" + "="*80)
    print("ALL JSON FILES PROCESSED. GENERATING FINAL PROMPTS.")
    print("="*80)

    # --- Call Gemini and Display Final Output ---
    print("\n" + "="*80)
    print("CALLING GEMINI API AND DISPLAYING FINAL OUTPUTS.")
    print("="*80)

    # Get overall rating and extract the value
    overall_rating_response = get_gemini_response(overall_rating_prompt)
    if overall_rating_response:
        # Use regex to find a floating-point number with two decimal places
        match = re.search(r'\d+\.\d{2}', overall_rating_response)
        extracted_rating = match.group(0) if match else "N/A"
        print(f"Overall Course Rating: {extracted_rating}")

    # Get and display learner perspective feedback
    learner_feedback_response = get_gemini_response(learner_prompt)
    if learner_feedback_response:
        print(learner_feedback_response)

    # Get and display instructor perspective feedback
    instructor_feedback_response = get_gemini_response(instructor_prompt)
    if instructor_feedback_response:
        print(instructor_feedback_response)

    final_results = {}
    if overall_rating_response:
        # Store only the extracted rating in the JSON output
        match = re.search(r'\d+\.\d{2}', overall_rating_response)
        extracted_rating = match.group(0) if match else "N/A"
        final_results["Overall Course Rating"] = extracted_rating

    if learner_feedback_response:
        final_results["Learner Perspective Feedback"] = learner_feedback_response
    if instructor_feedback_response:
        final_results["Instructor Feedback"] = instructor_feedback_response

    output_file = "UI_LEOA_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    print(f"\nAnalysis results have been saved to '{output_file}'.")
