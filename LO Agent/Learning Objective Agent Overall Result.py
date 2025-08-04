import json
import re
import os
import google.generativeai as genai

def extract_course_level_data(json_content):
    """
    Extracts scores and feedback from the 'Course Level Evaluation' section of a JSON.
    Handles different JSON structures. Returns a list of dictionaries, each containing
    'name', 'score', 'learner_feedback', and 'instructor_feedback'.
    """
    course_level_eval = json_content.get("Course Level Evaluation", {})
    extracted_entries = []

    # Check for the multi-score structure (like Quality Check Results.json)
    # This is identified by nested dictionaries with a "Value" key
    is_multi_score_structure = any(
        isinstance(course_level_eval.get(key), dict) and "Value" in course_level_eval[key]
        for key in course_level_eval
    )

    if is_multi_score_structure:
        for key, value_dict in course_level_eval.items():
            if isinstance(value_dict, dict) and "Value" in value_dict:
                sub_score = value_dict["Value"]
                sub_name = key.replace(' Score', '').replace('_', ' ').title() # E.g., "Linguistic Quality"
                
                learner_fb_text = ""
                instructor_fb_text = ""
                if "Detailed Insights" in value_dict:
                    insights = value_dict["Detailed Insights"]
                    learner_fb_key = next((k for k in insights if "learner perspective assessment" in k.lower()), None)
                    instructor_fb_key = next((k for k in insights if "instructor feedback" in k.lower()), None)

                    if learner_fb_key:
                        learner_fb_text = insights[learner_fb_key]
                    if instructor_fb_key:
                        instructor_fb_text = insights[instructor_fb_key]
                
                extracted_entries.append({
                    "name": sub_name,
                    "score": sub_score,
                    "learner_feedback": learner_fb_text,
                    "instructor_feedback": instructor_fb_text
                })
    else:
        # Handle single-score structure (e.g., Syllabus Validation Results.json)
        score_key = None
        score = "N/A" # Default score should be "N/A" if not found as a number
        for key, value in course_level_eval.items():
            if isinstance(value, (int, float)) and "score" in key.lower():
                score_key = key
                score = value
                break
            # Added check: If the value is a string "N/A" and it's a score key, confirm N/A
            elif isinstance(value, str) and value.lower() == "n/a" and "score" in key.lower():
                score_key = key
                score = "N/A"
                break
        

        learner_fb_keys = ["Course Learner Perspective Assessment", "Learner Perspective Assessment"]
        inst_fb_keys = ["Course Instructor Feedback", "Instructor Feedback"]

        l_fb = ""
        for k in learner_fb_keys:
            if k in course_level_eval:
                # If feedback is "N/A" string, treat as empty
                if isinstance(course_level_eval[k], str) and course_level_eval[k].lower() == "n/a":
                    l_fb = ""
                else:
                    l_fb = course_level_eval[k]
                break
        
        i_fb = ""
        for k in inst_fb_keys:
            if k in course_level_eval:
                # If feedback is "N/A" string, treat as empty
                if isinstance(course_level_eval[k], str) and course_level_eval[k].lower() == "n/a":
                    i_fb = ""
                else:
                    i_fb = course_level_eval[k]
                break
        
        extracted_entries.append({
            "name": "Placeholder",
            "score": score,
            "learner_feedback": l_fb,
            "instructor_feedback": i_fb
        })

    return extracted_entries

def calculate_overall_rating_prompt_with_prereq_function(scores_data):
    print("Executing: calculate_overall_rating_prompt_with_prereq_function")
    """
    Generates a detailed prompt for an AI model to calculate an overall course rating
    including the prerequisite section.
    """
    scores_list_str = "\n".join([f"- {item['name']}: {item['score']:.2f}/5.0" for item in scores_data if item['score'] != "N/A"])

    if not scores_list_str:
        return "No valid scores were provided for overall rating calculation."

    return f"""
Persona: You are a distinguished member of an Academic Oversight Committee, tasked with performing a rigorous, data-driven assessment of a course's overall quality. Your role demands impartiality, analytical precision, and the ability to synthesize diverse validation metrics into a single, authoritative rating.

Your primary objective is to calculate an overall course quality rating on a scale of 1.0 to 5.0, rounded to two decimal places. This rating must be a direct reflection of the comprehensive quality indicated by several key validation scores, each representing a crucial dimension of the course's design and content. Your final output will also include a concise justification of this rating, outlining how the individual scores contributed to the final assessment through a holistic evaluation.

**Evaluation Methodology (How the overall rating is derived through qualitative synthesis):**

* **Syllabus Validation Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric quantifies the effectiveness and quality of the course's syllabus. It assesses how well the stated learning objectives align with the course content, the logical flow and coherence of topics, and the comprehensiveness of the curriculum as presented in the syllabus documents.
    * **Scoring:** 1.0 (Indicates severe misalignment, significant gaps, or a highly incoherent syllabus structure, likely leading to learner confusion) to 5.0 (Signifies exceptional alignment, comprehensive coverage, and a perfectly logical, clear, and robust syllabus design).

* **Prerequisite Validation Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric evaluates the suitability and integration of prerequisite knowledge required for the course. It assesses whether the assumed foundational knowledge is clearly communicated, adequately prepared for, and seamlessly integrated into the initial course modules, ensuring learners have the necessary baseline for success.
    * **Scoring:** 1.0 (Suggests a significant mismatch between assumed and required knowledge, or poor reinforcement of prerequisites, creating substantial learning barriers) to 5.0 (Represents perfectly aligned and strategically reinforced prerequisites that smoothly transition learners into new material).

* **Learning Objective Validation Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric measures the quality, clarity, attainability, and internal coherence of the course's stated learning objectives. It ensures that the objectives are specific, measurable, achievable, relevant, time-bound (SMART), and collectively form a unified and logical set of educational goals for the learner.
    * **Scoring:** 1.0 (Points to objectives that are vague, unattainable, or disconnected, severely hindering learner focus and assessment) to 5.0 (Signifies learning objectives that are exceptionally clear, perfectly aligned, attainable, and form a highly coherent and effective roadmap for learning).

* **Linguistic Quality Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses the grammatical correctness, clarity, and overall precision of the language used throughout the course materials. It ensures effective communication and minimizes ambiguity.
    * **Scoring:** 1.0 (Denotes significant grammatical errors or unclear phrasing that hinders comprehension) to 5.0 (Signifies impeccable grammar, precise vocabulary, and crystal-clear expression).

* **Fluency Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric evaluates the natural flow, coherence, and logical progression of ideas within the course content. It assesses how smoothly transitions are made between topics and concepts.
    * **Scoring:** 1.0 (Indicates disjointed ideas and poor transitions, making the content difficult to follow) to 5.0 (Represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience).

* **Semantic Coverage Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric measures the completeness and depth of topic coverage, ensuring that all necessary concepts are introduced and explained adequately within their context. It assesses whether the content fully addresses the scope implied by the learning objectives.
    * **Scoring:** 1.0 (Suggests significant gaps in content or superficial coverage of key topics) to 5.0 (Signifies comprehensive and well-contextualized coverage of all relevant concepts).

* **Readability Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses how easy the course materials are to read and understand for the target audience. It considers factors such as sentence length, vocabulary complexity, and overall textual structure.
    * **Scoring:** 1.0 (Implies highly complex or dense text that is challenging to read and comprehend) to 5.0 (Represents clear, concise, and easily digestible text suitable for effective learning).

**Underlying Quality Observations (These are the precise numerical inputs for your assessment):**
{scores_list_str}

**Overall Rating Determination:**
-   The overall course quality rating will be determined by a **holistic assessment** of all the individual scores provided above. This is not a simple arithmetic calculation, but rather a reasoned judgment based on the strengths and weaknesses indicated by each dimension.
-   Consider how each score contributes to the overall effectiveness and quality of the course. A low score in one critical area might significantly impact the overall rating, even if other scores are high. Conversely, consistently high scores across all dimensions should lead to an excellent overall rating.
-   The final overall rating must be presented as a floating-point number, rounded to exactly two decimal places.

**Expected Response Guidelines:**
* **Perspective:** From the authoritative, analytical point of view of an Academic Oversight Committee member.
* **Tone:** Formal, impartial, and highly data-driven.
* **Style:** Concise and direct. Avoid conversational fillers or overly flowery language.
* **Focus:** Directly state the calculated overall rating and provide a clear, brief justification that references the contributions of the underlying individual scores and their collective impact on the final assessment through a qualitative assessment.
* **Avoid:** Any mention of specific mathematical formulas (e.g., "average," "sum divided by") or the inclusion of qualitative feedback from learners/instructors (as that is handled in a separate prompt).

**Prompt Instruction:**
Given the underlying quality observations of all provided scores, determine the overall course quality rating out of 5.0 through a holistic assessment, and provide a brief, data-driven justification for it, explaining how the individual scores informed this qualitative judgment.

**Required Output Format:**
Overall Course Rating: [Your Assessed Rating out of 5.0 with two decimal points]
Justification: [Your detailed reasoning for the rating, based on the qualitative assessment of individual score contributions.]
"""

def calculate_overall_rating_prompt_without_prereq_function(scores_data):
    print("Executing: calculate_overall_rating_prompt_without_prereq_function")
    """
    Generates a detailed prompt for an AI model to calculate an overall course rating
    EXCLUDING the prerequisite section.
    """
    scores_list_str = "\n".join([f"- {item['name']}: {item['score']:.2f}/5.0" for item in scores_data if item['score'] != "N/A"])

    if not scores_list_str:
        return "No valid scores were provided for overall rating calculation."

    return f"""
Persona: You are a distinguished member of an Academic Oversight Committee, tasked with performing a rigorous, data-driven assessment of a course's overall quality. Your role demands impartiality, analytical precision, and the ability to synthesize diverse validation metrics into a single, authoritative rating.

Your primary objective is to calculate an overall course quality rating on a scale of 1.0 to 5.0, rounded to two decimal places. This rating must be a direct reflection of the comprehensive quality indicated by several key validation scores, each representing a crucial dimension of the course's design and content. Your final output will also include a concise justification of this rating, outlining how the individual scores contributed to the final assessment through a holistic evaluation.

**Evaluation Methodology (How the overall rating is derived through qualitative synthesis):**

* **Syllabus Validation Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric quantifies the effectiveness and quality of the course's syllabus. It assesses how well the stated learning objectives align with the course content, the logical flow and coherence of topics, and the comprehensiveness of the curriculum as presented in the syllabus documents.
    * **Scoring:** 1.0 (Indicates severe misalignment, significant gaps, or a highly incoherent syllabus structure, likely leading to learner confusion) to 5.0 (Signifies exceptional alignment, comprehensive coverage, and a perfectly logical, clear, and robust syllabus design).

* **Learning Objective Validation Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric measures the quality, clarity, attainability, and internal coherence of the course's stated learning objectives. It ensures that the objectives are specific, measurable, achievable, relevant, time-bound (SMART), and collectively form a unified and logical set of educational goals for the learner.
    * **Scoring:** 1.0 (Points to objectives that are vague, unattainable, or disconnected, severely hindering learner focus and assessment) to 5.0 (Signifies learning objectives that are exceptionally clear, perfectly aligned, attainable, and form a highly coherent and effective roadmap for learning).

* **Linguistic Quality Score (Score Range: 1.0-5.0):
    * **Definition:** This metric assesses the grammatical correctness, clarity, and overall precision of the language used throughout the course materials. It ensures effective communication and minimizes ambiguity.
    * **Scoring:** 1.0 (Denotes significant grammatical errors or unclear phrasing that hinders comprehension) to 5.0 (Signifies impeccable grammar, precise vocabulary, and crystal-clear expression).

* **Fluency Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric evaluates the natural flow, coherence, and logical progression of ideas within the course content. It assesses how smoothly transitions are made between topics and concepts.
    * **Scoring:** 1.0 (Indicates disjointed ideas and poor transitions, making the content difficult to follow) to 5.0 (Represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience).

* **Semantic Coverage Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric measures the completeness and depth of topic coverage, ensuring that all necessary concepts are introduced and explained adequately within their context. It assesses whether the content fully addresses the scope implied by the learning objectives.
    * **Scoring:** 1.0 (Suggests significant gaps in content or superficial coverage of key topics) to 5.0 (Signifies comprehensive and well-contextualized coverage of all relevant concepts).

* **Readability Score (Score Range: 1.0-5.0):**
    * **Definition:** This metric assesses how easy the course materials are to read and understand for the target audience. It considers factors such as sentence length, vocabulary complexity, and overall textual structure.
    * **Scoring:** 1.0 (Implies highly complex or dense text that is challenging to read and comprehend) to 5.0 (Represents clear, concise, and easily digestible text suitable for effective learning).

**Underlying Quality Observations (These are the precise numerical inputs for your assessment):**
{scores_list_str}

**Overall Rating Determination:**
-   The overall course quality rating will be determined by a **holistic assessment** of all the individual scores provided above. This is not a simple arithmetic calculation, but rather a reasoned judgment based on the strengths and weaknesses indicated by each dimension.
-   Consider how each score contributes to the overall effectiveness and quality of the course. A low score in one critical area might significantly impact the overall rating, even if other scores are high. Conversely, consistently high scores across all dimensions should lead to an excellent overall rating.
-   The final overall rating must be presented as a floating-point number, rounded to exactly two decimal places.

**Expected Response Guidelines:**
* **Perspective:** From the authoritative, analytical point of view of an Academic Oversight Committee member.
* **Tone:** Formal, impartial, and highly data-driven.
* **Style:** Concise and direct. Avoid conversational fillers or overly flowery language.
* **Focus:** Directly state the calculated overall rating and provide a clear, brief justification that references the contributions of the underlying individual scores and their collective impact on the final assessment through a qualitative assessment.
* **Avoid:** Any mention of specific mathematical formulas (e.g., "average," "sum divided by") or the inclusion of qualitative feedback from learners/instructors (as that is handled in a separate prompt).

**Prompt Instruction:**
Given the underlying quality observations of all provided scores, determine the overall course quality rating out of 5.0 through a holistic assessment, and provide a brief, data-driven justification for it, explaining how the individual scores informed this qualitative judgment.

**Required Output Format:**
Overall Course Rating: [Your Assessed Rating out of 5.0 with two decimal points]
Justification: [Your detailed reasoning for the rating, based on the qualitative assessment of individual score contributions.]
"""

def generate_learner_perspective_summary_prompt_with_prereq_function(feedback_data):
    print("Executing: generate_learner_perspective_summary_prompt_with_prereq_function")
    """
    Generates a detailed prompt for an AI model to summarize learner feedback,
    including the prerequisite section.
    """
    feedback_observations_str = "\n".join([
        f"- From {item['source']} Learner Feedback (Score: {item['score']:.2f}/5.0): \"{item['feedback_text'].strip()}\""
        for item in feedback_data if item['feedback_text'].strip() and item['score'] != "N/A"
    ])

    if not feedback_observations_str:
        return "No valid learner feedback observations were provided."

    mandated_areas_str = """
1.  Syllabus Effectiveness and Guidance
2.  Prerequisite Readiness and Smooth Integration
3.  Clarity and Rigor of Learning Objectives
4.  Overall Language Quality (implicitly covering linguistic quality, fluency, and readability of materials)
"""
    return f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality and cohesiveness. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's various interconnected elements (like its syllabus, prerequisites, learning objectives, and language quality) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. Your output should directly convey your feelings and observations about the course's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes, referencing each metric's specific mechanism):**
These categories represent the various facets of my learning journey that contribute to my overall perception. My reflection is implicitly informed by these distinct sources of feedback.

* **Syllabus Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This captures my direct experience with how well the course syllabus guided my learning, whether the content felt well-structured, clear, and pedagogically sound. It assesses the coherence of the overall journey, considering alignment with learning objectives and course goals, logical flow of topics, and clarity of cognitive challenge and topic diversity as presented in the syllabus.
    * **Scoring Context:** A lower score here would indicate a confusing or misaligned syllabus that severely hindered my learning direction, while a higher score signifies an exceptionally clear, well-structured, and perfectly guiding syllabus experience.

* **Prerequisite Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This reflects my feelings about the assumed prior knowledge for the course, whether it was adequately prepared for, and how smoothly the foundational concepts integrated into the main topics, ensuring I was neither overwhelmed nor underprepared. This score considers how well module-specific prerequisite knowledge aligns with the course's stated prerequisites, based on semantic closeness, logical entailment, and overall structural alignment.
    * **Scoring Context:** A lower score suggests significant gaps or poor integration of assumed knowledge, creating substantial early learning barriers, whereas a higher score means prerequisites were perfectly aligned and ensured a seamless start.

* **Learning Objective Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This conveys my perception of the clarity, relevance, measurability, and cognitive alignment of the course's stated goals, and how effectively the content helped me achieve those objectives. This is determined by how well the objectives align with extracted course concepts and appropriate Bloom’s Taxonomy levels, and how well the original objectives match clearly generated measurable objectives.
    * **Scoring Context:** A lower score points to objectives that felt vague, unattainable, or disconnected, severely hindering my focus, while a higher score signifies exceptionally clear, relevant, and attainable objectives that truly guided my learning effectively.

* **Linguistic Quality Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This summarizes my experience with the grammatical correctness, clarity, and precision of the language used throughout the course materials, such as lecture transcripts. It assesses grammar, spelling, and effective communication.
    * **Scoring Context:** A lower score denotes significant errors or unclear phrasing that hindered comprehension, whereas a higher score signifies impeccable grammar and crystal-clear expression that enhanced understanding.

* **Fluency Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This reflects my perception of how smoothly ideas flow and connect within the course content, particularly in lecture transcripts. It assesses how well transitions are made between topics and concepts.
    * **Scoring Context:** A lower score indicates disjointed ideas and poor transitions, making the content difficult to follow, while a higher score represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience.

* **Semantic Coverage Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This conveys my understanding of how completely and deeply topics were covered within the course materials, ensuring all necessary concepts are introduced and explained adequately within their context.
    * **Scoring Context:** A lower score suggests significant gaps in content or superficial coverage of key topics, leaving me with incomplete understanding, while a higher score signifies comprehensive and well-contextualized coverage, ensuring thorough learning.

* **Readability Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This summarizes my experience with how easy the course materials (e.g., lecture transcripts) were to read and understand for the target audience. It considers factors such as sentence length, vocabulary complexity, and overall textual structure, and how well the language aligns with the expected learner level.
    * **Scoring Context:** A lower score implies highly complex or dense text that is challenging to read and comprehend, whereas a higher score represents clear, concise, and easily digestible text suitable for effective learning.

**Underlying Quality Observations (These are the direct textual observations influencing my perceived experience, along with their associated quality scores. Use these scores to gauge the *severity* or *strength* of the feedback, but DO NOT explicitly state their technical names, sources, or numerical values in my reflection):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 4 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated reflection **MUST be a maximum of 4 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly convey observations related to the learner's experience regarding:**
{mandated_areas_str}
Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using connecting phrases to maintain flow. Each sentence should contribute meaningfully to the overall assessment, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** A highly cohesive and evaluative narrative of **2–4 sentences ONLY**. This narrative should build towards an overarching impression of the learning experience, strategically incorporating observations for each mandated analytical area without sounding like a list. **Start directly with the content, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list feedback points.** Instead, **holistically analyze and integrate** all underlying feedback to form a unified assessment of the course's overall effectiveness, clarity, and engagement. The narrative should seamlessly combine observations about how the various course elements, interpreted through the context of their specific methodologies, collectively shaped the experience, subtly emphasizing strengths or weaknesses based on associated scores. For instance, a low score accompanying a feedback point should imply a more significant perceived deficiency that contributes to the overall feeling.
* **Avoid:** Technical jargon, generic introductions, directly mentioning the metric names or numerical scores, or explicit reference to the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various learner feedback sources, including their associated quality scores, **conduct a thorough analysis to form a unified learner-style reflection** on the overall quality and cohesiveness of the course. **Integrate all insights** into a coherent narrative that simulates how the learner *perceived* the course's structure, content, and clarity, referencing the specific methodologies behind each evaluated aspect. **YOUR 2-4 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE LEARNER'S EXPERIENCE REGARDING THE SYLLABUS, PREREQUISITES, LEARNING OBJECTIVES, AND THE OVERALL LANGUAGE QUALITY.** The final sentence must be a concise, overarching assessment of the course's effectiveness from a learner's vantage point, acting as a clear key takeaway.

**Required Output Format:**
**Final Learner Perspective Feedback:**
[Your deeply analyzed and integrated summary of all learner feedback, in learner's voice, **STRICTLY 2-4 sentences**, no first-person pronouns, focusing on an overall assessment rather than just synthesis, and **analyzing each analytical area very clearly** including the learner's experience with the Syllabus, Prerequisites, Learning Objectives, and Language Quality. **Start directly with the content.**]
"""

def generate_learner_perspective_summary_prompt_without_prereq_function(feedback_data):
    print("Executing: generate_learner_perspective_summary_prompt_without_prereq_function")
    """
    Generates a detailed prompt for an AI model to summarize learner feedback,
    EXCLUDING the prerequisite section.
    """
    feedback_observations_str = "\n".join([
        f"- From {item['source']} Learner Feedback (Score: {item['score']:.2f}/5.0): \"{item['feedback_text'].strip()}\""
        for item in feedback_data if item['feedback_text'].strip() and item['score'] != "N/A"
    ])

    if not feedback_observations_str:
        return "No valid learner feedback observations were provided."

    mandated_areas_str = """
1.  Syllabus Effectiveness and Guidance
2.  Clarity and Rigor of Learning Objectives
3.  Overall Language Quality (implicitly covering linguistic quality, fluency, and readability of materials)
"""
    return f"""
Persona: You are simulating a learner's direct experience and perception of a course's overall quality and cohesiveness. Your goal is to provide a personal reflection, as if you were the student who just completed the entire course.

This reflection focuses on how effectively the course's various interconnected elements (like its syllabus, learning objectives, and language quality) came together to deliver a cohesive, accessible, and ultimately enriching learning experience for me. Your output should directly convey your feelings and observations about the course's effectiveness.

**Evaluation Methodology (How my experience is being assessed behind the scenes, referencing each metric's specific mechanism):**
These categories represent the various facets of my learning journey that contribute to my overall perception. My reflection is implicitly informed by these distinct sources of feedback.

* **Syllabus Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This captures my direct experience with how well the course syllabus guided my learning, whether the content felt well-structured, clear, and pedagogically sound. It assesses the coherence of the overall journey, considering alignment with learning objectives and course goals, logical flow of topics, and clarity of cognitive challenge and topic diversity as presented in the syllabus.
    * **Scoring Context:** A lower score here would indicate a confusing or misaligned syllabus that severely hindered my learning direction, while a higher score signifies an exceptionally clear, well-structured, and perfectly guiding syllabus experience.

* **Learning Objective Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This conveys my perception of the clarity, relevance, measurability, and cognitive alignment of the course's stated goals, and how effectively the content helped me achieve those objectives. This is determined by how well the objectives align with extracted course concepts and appropriate Bloom’s Taxonomy levels, and how well the original objectives match clearly generated measurable objectives.
    * **Scoring Context:** A lower score points to objectives that felt vague, unattainable, or disconnected, severely hindering my focus, while a higher score signifies exceptionally clear, relevant, and attainable objectives that truly guided my learning effectively.

* **Linguistic Quality Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This summarizes my experience with the grammatical correctness, clarity, and precision of the language used throughout the course materials, such as lecture transcripts. It assesses grammar, spelling, and effective communication.
    * **Scoring Context:** A lower score denotes significant errors or unclear phrasing that hindered comprehension, whereas a higher score signifies impeccable grammar and crystal-clear expression that enhanced understanding.

* **Fluency Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This reflects my perception of how smoothly ideas flow and connect within the course content, particularly in lecture transcripts. It assesses how well transitions are made between topics and concepts.
    * **Scoring Context:** A lower score indicates disjointed ideas and poor transitions, making the content difficult to follow, while a higher score represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience.

* **Semantic Coverage Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This conveys my understanding of how completely and deeply topics were covered within the course materials, ensuring all necessary concepts are introduced and explained adequately within their context.
    * **Scoring Context:** A lower score suggests significant gaps in content or superficial coverage of key topics, leaving me with incomplete understanding, while a higher score signifies comprehensive and well-contextualized coverage, ensuring thorough learning.

* **Readability Learner Perspective (Score Range: 1.0-5.0):**
    * **Definition:** This summarizes my experience with how easy the course materials (e.g., lecture transcripts) were to read and understand for the target audience. It considers factors such as sentence length, vocabulary complexity, and overall textual structure, and how well the language aligns with the expected learner level.
    * **Scoring Context:** A lower score implies highly complex or dense text that is challenging to read and comprehend, whereas a higher score represents clear, concise, and easily digestible text suitable for effective learning.

**Underlying Quality Observations (These are the direct textual observations influencing my perceived experience, along with their associated quality scores. Use these scores to gauge the *severity* or *strength* of the feedback, but DO NOT explicitly state their technical names, sources, or numerical values in my reflection):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 4 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated reflection **MUST be a maximum of 4 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly convey observations related to the learner's experience regarding:**
{mandated_areas_str}
Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using connecting phrases to maintain flow. Each sentence should contribute meaningfully to the overall assessment, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the personal, simulated point of view of the learner.
* **Tone:** Reflective, authentic, and concise. **Strictly use passive voice or objective statements; avoid any first-person pronouns (I, my, me).**
* **Style:** A highly cohesive and evaluative narrative of **2–4 sentences ONLY**. This narrative should build towards an overarching impression of the learning experience, strategically incorporating observations for each mandated analytical area without sounding like a list. **Start directly with the content, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list feedback points.** Instead, **holistically analyze and integrate** all underlying feedback to form a unified assessment of the course's overall effectiveness, clarity, and engagement. The narrative should seamlessly combine observations about how the various course elements, interpreted through the context of their specific methodologies, collectively shaped the experience, subtly emphasizing strengths or weaknesses based on associated scores. For instance, a low score accompanying a feedback point should imply a more significant perceived deficiency that contributes to the overall feeling.
* **Avoid:** Technical jargon, generic introductions, directly mentioning the metric names or numerical scores, or explicit reference to the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various learner feedback sources, including their associated quality scores, **conduct a thorough analysis to form a unified learner-style reflection** on the overall quality and cohesiveness of the course. **Integrate all insights** into a coherent narrative that simulates how the learner *perceived* the course's structure, content, and clarity, referencing the specific methodologies behind each evaluated aspect. **YOUR 2-4 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE LEARNER'S EXPERIENCE REGARDING THE SYLLABUS, LEARNING OBJECTIVES, AND THE OVERALL LANGUAGE QUALITY.** The final sentence must be a concise, overarching assessment of the course's effectiveness from a learner's vantage point, acting as a clear key takeaway.

**Required Output Format:**
**Final Learner Perspective Feedback:**
[Your deeply analyzed and integrated summary of all learner feedback, in learner's voice, **STRICTLY 2-4 sentences**, no first-person pronouns, focusing on an overall assessment rather than just synthesis, and **analyzing each analytical area very clearly** including the learner's experience with the Syllabus, Learning Objectives, and Language Quality. **Start directly with the content.**]
"""

def generate_instructor_feedback_summary_prompt_with_prereq_function(feedback_data):
    print("Executing: generate_instructor_feedback_summary_prompt_with_prereq_function")
    """
    Generates a detailed prompt for an AI model to summarize instructor feedback,
    including the prerequisite section.
    """
    feedback_observations_str = "\n".join([
        f"- From {item['source']} Instructor Feedback (Score: {item['score']:.2f}/5.0): \"{item['feedback_text'].strip()}\""
        for item in feedback_data if item['feedback_text'].strip() and item['score'] != "N/A"
    ])

    if not feedback_observations_str:
        return "No valid instructor feedback observations were provided."

    mandated_areas_str = """
1.  Syllabus Structure and Alignment
2.  Prerequisite Effectiveness and Integration
3.  Learning Objective Rigor and Coherence
4.  Overall Language Quality (implicitly covering linguistic quality, fluency, and readability of materials)
"""
    return f"""
Persona: You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this course's design and its overall pedagogical effectiveness. Your insights aim to optimize the course for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the course by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall educational goals. Your aim is to pinpoint specific design strengths across the course's various components and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective, leveraging the insights from a detailed review system.

**Evaluation Metrics & Scoring Principles (for your understanding of feedback sources, referencing each metric's specific mechanism):**
These categories represent the various facets of the course design that have been assessed by instructors. Your consolidated feedback will synthesize insights from these areas.

* **Syllabus Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides expert pedagogical insights into the course's content alignment, consistency across modules, and the depth and strategic sequencing of topics as outlined in the syllabus. It reflects the overall quality, structure, clarity, and pedagogical soundness of the syllabus, integrating analyses of learning objective-syllabus alignment, course-syllabus alignment, cognitive complexity (Bloom's Taxonomy), and topic diversity.
    * **Scoring Context:** A lower score here indicates a highly confusing or misaligned syllabus that requires urgent revision, while a higher score signifies an exceptionally clear and robust syllabus design.

* **Prerequisite Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This offers a professional assessment of the effectiveness of prerequisite requirements, their explicit communication, and strategies for their reinforcement and seamless integration into the course flow, ensuring learners are neither overwhelmed nor underprepared. This score is derived from evaluating how well module-specific prerequisites (generated based on module name, syllabus, and learning objectives) align with course-level prerequisites through semantic closeness, logical entailment, and structural completeness.
    * **Scoring Context:** A lower score suggests a significant mismatch or poor reinforcement of assumed knowledge, creating substantial learning barriers that need immediate attention, whereas a higher score means prerequisites were perfectly aligned and ensured a seamless start.

* **Learning Objective Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This delivers analytical recommendations concerning the coherence, clarity, measurability, and logical progression of the course's learning objectives, and their alignment with the overarching educational mission and intended cognitive depth. This evaluation involves extracting high-level course concepts, assigning Bloom's Taxonomy levels to syllabus points, generating clear and measurable objectives, and then comparing these generated objectives with the original ones based on semantic closeness, logical alignment, structural coverage, and Bloom-level match.
    * **Scoring Context:** A lower score points to objectives that are vague, unattainable, or disconnected, severely hindering learner focus and assessment, and thus requiring significant re-evaluation and instructional rigor improvements, while a higher score signifies exceptionally clear, relevant, and attainable objectives that truly guide learning effectively.

* **Linguistic Quality Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This assesses the grammatical precision, clarity, and overall correctness of the course's language used throughout the course materials, particularly lecture transcripts. This assessment includes checks for grammar and spelling, and measures of sentence flow.
    * **Scoring Context:** A lower score denotes significant grammatical errors or unclear phrasing that hinders comprehension and requires immediate correction, whereas a higher score signifies impeccable grammar and crystal-clear expression that optimizes communication.

* **Fluency Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides expert analysis on the coherence, logical progression, and smooth transitions of content within the course modules, particularly in lecture transcripts.
    * **Scoring Context:** A lower score indicates disjointed ideas and poor transitions, making the content difficult to follow and necessitating major structural review, while a higher score represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience.

* **Semantic Coverage Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This offers recommendations for optimizing the completeness, depth, and contextual relevance of subject matter coverage within the course materials, ensuring all necessary concepts are introduced and explained adequately.
    * **Scoring Context:** A lower score suggests significant gaps in content or superficial coverage of key topics, requiring substantial expansion, while a higher score signifies comprehensive and well-contextualized coverage of all relevant concepts.

* **Readability Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides guidance on improving the ease of comprehension and accessibility of course materials (e.g., lecture transcripts) through linguistic simplification and structural adjustments for the target audience. This assessment considers factors such as sentence length, vocabulary complexity, overall textual structure, and language alignment with the expected learner level.
    * **Scoring Context:** A lower score implies highly complex or dense text that is challenging to read and comprehend for the target audience, necessitating significant simplification, while a higher score represents clear, concise, and easily digestible text suitable for effective learning.

**Underlying Quality Observations (These are the precise analytical feedback entries informing your expert recommendations, along with their associated quality scores. Use these scores to gauge the *urgency* or *impact* of the feedback, but DO NOT state their technical names or sources explicitly in your feedback):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 4 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated pedagogical assessment **MUST be a maximum of 4 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly provide recommendations or observations related to the design quality of:**
{mandated_areas_str}
Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using strong action-oriented phrasing to maintain flow. Each sentence should contribute meaningfully to the overall assessment and recommendation, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise, integrated, and concise assessment of **2–4 sentences ONLY**. This assessment should flow logically, presenting a unified view of strengths and areas for improvement, strategically incorporating observations/recommendations for each mandated analytical area without sounding like a list. **Start directly with the assessment/recommendations, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list individual feedback points.** Instead, **conduct a thorough analytical review of all underlying observations** to form a cohesive assessment. Clearly articulate overarching design strengths and offer **concrete, strategic, and prioritized improvements** to the course's pedagogical efficacy, directly leveraging the insights from the underlying feedback, interpreted through the context of their specific methodologies. Let the numerical scores heavily influence the prioritization and emphasis of recommendations; lower scores should directly translate to higher urgency for suggested improvements.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores from the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various instructor feedback sources, including their associated quality scores, **provide a comprehensive pedagogical assessment and actionable recommendations** for the course's overall design and effectiveness. **Analyze and integrate all provided information** to articulate a unified view of areas for improvement and key strengths, based on the detailed methodology of each metric. **YOUR 2-4 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE DESIGN QUALITY OF THE SYLLABUS, THE EFFECTIVENESS OF PREREQUISITES, THE RIGOR OF LEARNING OBJECTIVES, AND THE OVERALL QUALITY OF COURSE LANGUAGE.** The final sentence must unequivocally outline a **single, prioritized, and overarching action item** for immediate course optimization, derived from the most impactful feedback.

**Required Output Format:**
**Final Instructor Feedback:**
[Your deeply analyzed, integrated, and action-oriented pedagogical assessment, presented in a consultant's voice, **STRICTLY 2-4 sentences**, clearly outlining key strengths, areas for improvement, and a single prioritized action item, and **analyzing each analytical area very clearly** including the Syllabus, Prerequisites, Learning Objectives, and Language Quality aspects. **Start directly with the content.**]
"""

def generate_instructor_feedback_summary_prompt_without_prereq_function(feedback_data):
    print("Executing: generate_instructor_feedback_summary_prompt_without_prereq_function")
    """
    Generates a detailed prompt for an AI model to summarize instructor feedback,
    EXCLUDING the prerequisite section.
    """
    feedback_observations_str = "\n".join([
        f"- From {item['source']} Instructor Feedback (Score: {item['score']:.2f}/5.0): \"{item['feedback_text'].strip()}\""
        for item in feedback_data if item['feedback_text'].strip() and item['score'] != "N/A"
    ])

    if not feedback_observations_str:
        return "No valid instructor feedback observations were provided."

    mandated_areas_str = """
1.  Syllabus Structure and Alignment
2.  Learning Objective Rigor and Coherence
3.  Overall Language Quality (implicitly covering linguistic quality, fluency, and readability of materials)
"""
    return f"""
Persona: You are an expert pedagogical consultant and curriculum design specialist. Your core function is to provide the instructor with incisive, data-driven feedback on this course's design and its overall pedagogical effectiveness. Your insights aim to optimize the course for maximum learning impact and seamless integration within the broader curriculum.

This assessment serves to validate the course by systematically dissecting its internal consistency, its fidelity in delivering advertised learning outcomes, and its strategic contribution to the overall educational goals. Your aim is to pinpoint specific design strengths across the course's various components and identify concrete, actionable areas for improvement that enhance instructional robustness and student success, directly from a validation perspective, leveraging the insights from a detailed review system.

**Evaluation Metrics & Scoring Principles (for your understanding of feedback sources, referencing each metric's specific mechanism):**
These categories represent the various facets of the course design that have been assessed by instructors. Your consolidated feedback will synthesize insights from these areas.

* **Syllabus Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides expert pedagogical insights into the course's content alignment, consistency across modules, and the depth and strategic sequencing of topics as outlined in the syllabus. It reflects the overall quality, structure, clarity, and pedagogical soundness of the syllabus, integrating analyses of learning objective-syllabus alignment, course-syllabus alignment, cognitive complexity (Bloom's Taxonomy), and topic diversity.
    * **Scoring Context:** A lower score here indicates a highly confusing or misaligned syllabus that requires urgent revision, while a higher score signifies an exceptionally clear and robust syllabus design.

* **Learning Objective Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This delivers analytical recommendations concerning the coherence, clarity, measurability, and logical progression of the course's learning objectives, and their alignment with the overarching educational mission and intended cognitive depth. This evaluation involves extracting high-level course concepts, assigning Bloom's Taxonomy levels to syllabus points, generating clear and measurable objectives, and then comparing these generated objectives with the original ones based on semantic closeness, logical alignment, structural coverage, and Bloom-level match.
    * **Scoring Context:** A lower score points to objectives that are vague, unattainable, or disconnected, severely hindering learner focus and assessment, and thus requiring significant re-evaluation and instructional rigor improvements, while a higher score signifies exceptionally clear, relevant, and attainable objectives that truly guide learning effectively.

* **Linguistic Quality Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This assesses the grammatical precision, clarity, and overall correctness of the course's language used throughout the course materials, particularly lecture transcripts. This assessment includes checks for grammar and spelling, and measures of sentence flow.
    * **Scoring Context:** A lower score denotes significant grammatical errors or unclear phrasing that hinders comprehension and requires immediate correction, whereas a higher score signifies impeccable grammar and crystal-clear expression that optimizes communication.

* **Fluency Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides expert analysis on the coherence, logical progression, and smooth transitions of content within the course modules, particularly in lecture transcripts.
    * **Scoring Context:** A lower score indicates disjointed ideas and poor transitions, making the content difficult to follow and necessitating major structural review, while a higher score represents excellent logical flow and seamless transitions, creating a highly cohesive learning experience.

* **Semantic Coverage Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This offers recommendations for optimizing the completeness, depth, and contextual relevance of subject matter coverage within the course materials, ensuring all necessary concepts are introduced and explained adequately.
    * **Scoring Context:** A lower score suggests significant gaps in content or superficial coverage of key topics, requiring substantial expansion, while a higher score signifies comprehensive and well-contextualized coverage of all relevant concepts.

* **Readability Instructor Feedback (Score Range: 1.0-5.0):**
    * **Definition:** This provides guidance on improving the ease of comprehension and accessibility of course materials (e.g., lecture transcripts) through linguistic simplification and structural adjustments for the target audience. This assessment considers factors such as sentence length, vocabulary complexity, overall textual structure, and language alignment with the expected learner level.
    * **Scoring Context:** A lower score implies highly complex or dense text that is challenging to read and comprehend for the target audience, necessitating significant simplification, while a higher score represents clear, concise, and easily digestible text suitable for effective learning.

**Underlying Quality Observations (These are the precise analytical feedback entries informing your expert recommendations, along with their associated quality scores. Use these scores to gauge the *urgency* or *impact* of the feedback, but DO NOT state their technical names or sources explicitly in your feedback):**
{feedback_observations_str}

**STRICT OUTPUT LENGTH & MANDATE: MAXIMUM 4 SENTENCES, COVERING ALL KEY ANALYTICAL AREAS**
Your generated pedagogical assessment **MUST be a maximum of 4 sentences, and a minimum of 2 sentences**. Within this strict limit, it **MUST explicitly provide recommendations or observations related to the design quality of:**
{mandated_areas_str}
Achieve this by concisely integrating insights, prioritizing the most impactful findings from each area, and using strong action-oriented phrasing to maintain flow. Each sentence should contribute meaningfully to the overall assessment and recommendation, ensuring no single analytical area is overlooked.

**Expected Response Guidelines:**
* **Perspective:** From the professional, data-driven point of view as a pedagogical consultant, directly addressing the instructor.
* **Tone:** Formal, analytical, authoritative, constructive, and highly action-oriented.
* **Style:** Deliver a highly precise, integrated, and concise assessment of **2–4 sentences ONLY**. This assessment should flow logically, presenting a unified view of strengths and areas for improvement, strategically incorporating observations/recommendations for each mandated analytical area without sounding like a list. **Start directly with the assessment/recommendations, no introductory phrases.**
* **Focus:** **Do not merely synthesize or list individual feedback points.** Instead, **conduct a thorough analytical review of all underlying observations** to form a cohesive assessment. Clearly articulate overarching design strengths and offer **concrete, strategic, and prioritized improvements** to the course's pedagogical efficacy, directly leveraging the insights from the underlying feedback, interpreted through the context of their specific methodologies. Let the numerical scores heavily influence the prioritization and emphasis of recommendations; lower scores should directly translate to higher urgency for suggested improvements.
* **Avoid:** Technical jargon not universally understood in pedagogical circles, or directly mentioning the metric names or numerical scores from the "Underlying Quality Observations" categories.

**Prompt Instruction:**
Given the underlying quality observations from various instructor feedback sources, including their associated quality scores, **provide a comprehensive pedagogical assessment and actionable recommendations** for the course's overall design and effectiveness. **Analyze and integrate all provided information** to articulate a unified view of areas for improvement and key strengths, based on the detailed methodology of each metric. **YOUR 2-4 SENTENCE RESPONSE MUST START DIRECTLY WITH THE ASSESSMENT AND EXPLICITLY ADDRESS THE DESIGN QUALITY OF THE SYLLABUS, THE RIGOR OF LEARNING OBJECTIVES, AND THE OVERALL QUALITY OF COURSE LANGUAGE.** The final sentence must unequivocally outline a **single, prioritized, and overarching action item** for immediate course optimization, derived from the most impactful feedback.

**Required Output Format:**
**Final Instructor Feedback:**
[Your deeply analyzed, integrated, and action-oriented pedagogical assessment, presented in a consultant's voice, **STRICTLY 2-4 sentences**, clearly outlining key strengths, areas for improvement, and a single prioritized action item, and **analyzing each analytical area very clearly** including the Syllabus, Learning Objectives, and Language Quality aspects. **Start directly with the content.**]
"""

def process_course_evaluations_from_files(file_paths):
    """
    Processes course evaluation JSON files from provided paths to extract scores and feedback,
    and then generates prompts for overall rating and feedback summary.
    Handles cases where prerequisite data might be missing or score is "N/A" by selecting
    the appropriate prompt function.

    Args:
        file_paths (list of str): A list of file paths to the JSON files.

    Returns:
        tuple: A tuple containing:
               - str: The prompt for calculating the overall rating.
               - str: The prompt for generating learner feedback summaries.
               - str: The prompt for generating instructor feedback summaries.
               - dict: A dictionary containing extracted data for verification (optional).
    """
    extracted_scores = []
    all_learner_feedback_entries = []
    all_instructor_feedback_entries = []
    
    # Flag to track if prerequisite score is N/A
    prerequisites_are_na = False 

    extracted_raw_data = {} # For debugging/verification

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found at path: {file_path}. Skipping.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_json_content = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
            continue

        entries_from_file = extract_course_level_data(full_json_content)
        
        file_name_short = os.path.basename(file_path).replace('.json', '').replace('_', ' ').title()

        # Check specifically for Prerequisite Validation and "N/A" score
        if "Prerequisites Validation Results" in file_name_short and entries_from_file and entries_from_file[0]['score'] == "N/A":
            prerequisites_are_na = True
            print(f"Prerequisite Validation Results has an 'N/A' score. Prompts will be generated without prerequisite definitions.")
            # Do NOT 'continue' here, as we still want to process other files and
            # collect any valid scores/feedback from them.
            # The filtering of 'N/A' prerequisite score/feedback from lists
            # `extracted_scores`, `all_learner_feedback_entries`, `all_instructor_feedback_entries`
            # will be handled in the loop below.

        for entry in entries_from_file:
            entry_name_for_prompt = entry['name'] if entry['name'] != "Placeholder" else file_name_short
            
            # Only add to scores if the score is not "N/A"
            if entry['score'] != "N/A":
                extracted_scores.append({"name": entry_name_for_prompt, "score": entry['score']})
            
            # Only add feedback if the score is not "N/A" and feedback exists
            if entry['learner_feedback'].strip() and entry['score'] != "N/A":
                all_learner_feedback_entries.append({
                    "source": entry_name_for_prompt,
                    "feedback_text": entry['learner_feedback'],
                    "score": entry['score']
                })
            if entry['instructor_feedback'].strip() and entry['score'] != "N/A":
                all_instructor_feedback_entries.append({
                    "source": entry_name_for_prompt,
                    "feedback_text": entry['instructor_feedback'],
                    "score": entry['score']
                })
            
            # For raw data verification, keep track of individual entries (even N/A for logging)
            if file_name_short not in extracted_raw_data:
                extracted_raw_data[file_name_short] = []
            extracted_raw_data[file_name_short].append({
                "name": entry_name_for_prompt,
                "score": entry['score'],
                "learner_feedback": entry['learner_feedback'],
                "instructor_feedback": entry['instructor_feedback']
            })

    # Now, generate prompts based on the prerequisites_are_na flag
    if prerequisites_are_na:
        overall_rating_prompt = calculate_overall_rating_prompt_without_prereq_function(extracted_scores)
        learner_feedback_prompt = generate_learner_perspective_summary_prompt_without_prereq_function(all_learner_feedback_entries)
        instructor_feedback_prompt = generate_instructor_feedback_summary_prompt_without_prereq_function(all_instructor_feedback_entries)
    else:
        overall_rating_prompt = calculate_overall_rating_prompt_with_prereq_function(extracted_scores)
        learner_feedback_prompt = generate_learner_perspective_summary_prompt_with_prereq_function(all_learner_feedback_entries)
        instructor_feedback_prompt = generate_instructor_feedback_summary_prompt_with_prereq_function(all_instructor_feedback_entries)

    return overall_rating_prompt, learner_feedback_prompt, instructor_feedback_prompt, extracted_raw_data

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
    # IMPORTANT: Replace these with the actual paths to your JSON files on your system.
    json_file_paths = [
       
    ]

    # Step 1: Process the files to extract data and generate prompts
    overall_rating_prompt, learner_feedback_prompt, instructor_feedback_prompt, extracted_raw_data = \
        process_course_evaluations_from_files(json_file_paths)

    print("--- Extracted Raw Data (for verification) ---")
    for file_name, entries in extracted_raw_data.items():
        print(f"\nFile: {file_name}")
        for entry in entries:
            # Check if score is a number before formatting, otherwise print as is (e.g., "N/A")
            score_display = f"{entry['score']:.2f}" if isinstance(entry['score'], (int, float)) else entry['score']
            print(f"  Name: {entry['name']}, Score: {score_display}")
            # Uncomment the lines below if you want to see the raw feedback extracted from files
            # print(f"  Learner FB: {entry['learner_feedback']}")
            # print(f"  Instructor FB: {entry['instructor_feedback']}")

    # Step 2: Send prompts to Gemini API and print responses

    # Overall Course Rating
    print("\n" + "="*80)
    print("--- SENDING PROMPT FOR OVERALL COURSE RATING TO GEMINI ---")
    print("="*80)
    # print(overall_rating_prompt) # Uncomment to see the full prompt before sending

    # Only send the prompt if it's not the "No valid scores" message
    if overall_rating_prompt != "No valid scores were provided for overall rating calculation.":
        overall_rating_response = get_gemini_response(overall_rating_prompt)
        if overall_rating_response:
            print("\n--- GEMINI'S OVERALL RATING RESPONSE ---")
            print(overall_rating_response)
        else:
            print("Failed to get overall rating from Gemini.")
    else:
        print("\n--- SKIPPING OVERALL RATING PROMPT: No valid scores available. ---")


    # Learner Perspective Feedback Summary
    print("\n" + "="*80)
    print("--- SENDING PROMPT FOR LEARNER PERSPECTIVE FEEDBACK TO GEMINI ---")
    print("="*80)
    # print(learner_feedback_prompt) # Uncomment to see the full prompt before sending

    # Only send the prompt if it's not the "No valid feedback" message
    if learner_feedback_prompt != "No valid learner feedback observations were provided.":
        learner_feedback_response = get_gemini_response(learner_feedback_prompt)
        if learner_feedback_response:
            print("\n--- GEMINI'S LEARNER FEEDBACK SUMMARY ---")
            print(learner_feedback_response)
        else:
            print("Failed to get learner feedback summary from Gemini.")
    else:
        print("\n--- SKIPPING LEARNER FEEDBACK PROMPT: No valid learner feedback available. ---")


    # Instructor Feedback Summary
    print("\n" + "="*80)
    print("--- SENDING PROMPT FOR INSTRUCTOR FEEDBACK TO GEMINI ---")
    print("="*80)
    # print(instructor_feedback_prompt) # Uncomment to see the full prompt before sending

    # Only send the prompt if it's not the "No valid feedback" message
    if instructor_feedback_prompt != "No valid instructor feedback observations were provided.":
        instructor_feedback_response = get_gemini_response(instructor_feedback_prompt)
        if instructor_feedback_response:
            print("\n--- GEMINI'S INSTRUCTOR FEEDBACK SUMMARY ---")
            print(instructor_feedback_response)
        else:
            print("Failed to get instructor feedback summary from Gemini.")
    else:
        print("\n--- SKIPPING INSTRUCTOR FEEDBACK PROMPT: No valid instructor feedback available. ---")