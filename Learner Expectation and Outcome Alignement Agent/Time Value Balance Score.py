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
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import torch
import re

# Initialize GPU acceleration
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("No GPU available, using CPU")

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
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

def save_dict_to_json(data: dict, file_path: str):
    """
    Saves a dictionary to a JSON file.

    Parameters:
    - data: dict, the dictionary to save
    - file_path: str, path to the output .json file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_folder_name(folder_id, service_account_file):
    # Authenticate using service account file
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=['https://www.googleapis.com/auth/drive.metadata.readonly']
    )

    # Build the Drive v3 API service
    service = build('drive', 'v3', credentials=credentials)

    # Get the folder metadata
    folder = service.files().get(fileId=folder_id, fields='name').execute()

    return folder.get('name')

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

def clean_response(response: str) -> int:
    """
    Extracts a valid score (1-5) from an LLM response string.
    Falls back to 3 if no valid number is found.
    """
    # Try to find the first number between 1 and 5
    match = re.search(r"\b([1-5])\b", response)
    if match:
        return int(match.group(1))
    return 3  # Default to neutral if LLM goes rogue

def get_total_hours(commitment):
    weeks = int(re.search(r"(\d+)\s*weeks?", commitment).group(1))
    hours_per_week = [int(h) for h in re.findall(r"(\d+)-?(\d+)?\s*hours?/week", commitment)[0] if h]
    avg_hours = sum(hours_per_week) / len(hours_per_week)
    total_hours = avg_hours * weeks
    return total_hours

def get_tvb_score(course_info, difficulty_level, prerequisites, commitment, learning_objectives, syllabus):
    total_hours = get_total_hours(commitment)
    prompt = f"""
    You are tasked with evaluating the educational content value delivered by a course based on several key aspects of its structure and intended outcomes. The provided metadata for this course includes:
    Course Description: {course_info}
    Course Difficulty Level: {difficulty_level}
    Course Prerequisites: {prerequisites}
    Learning Objectives: {learning_objectives}
    Syllabus Outline: {syllabus}
    Taking all the information above into account, please analyze and estimate the overall value that this course offers to a learner in terms of its content quality, the relevance and clarity of its learning objectives, the appropriateness of prerequisites, the comprehensiveness and depth of the syllabus, and its alignment with the specified difficulty level. 
    Your assessment should reflect the extent to which the course is likely to meet learner expectations, support meaningful learning outcomes, and justify the investment of time required for completion.
    Rate the overall educational content value of the course using a single integer score on a scale from 1 to 5, where:
    1 -> indicates minimal value or poor alignment with learner needs
    2 -> indicates below-average value
    3 -> indicates moderate or average value
    4 -> indicates above-average value
    5 -> indicates exceptional value or strong alignment with learner needs and expectations
    Your output should be only this integer score (1 to 5).
    Note: This value score will be further utilized in combination with the total duration of the course to determine the time-value balance as part of a broader learner expectation and outcome alignment analysis.
    """
    value_score = clean_response(execute_prompt(prompt))
    value_per_hour = value_score / total_hours
    if value_per_hour < 0.3:
        final_score = 1
    elif value_per_hour < 0.6:
        final_score = 2
    elif value_per_hour < 1.0:
        final_score = 3
    elif value_per_hour < 1.5:
        final_score = 4
    else:
        final_score = 5
    return {"total_hours": total_hours, "value_score": value_score, "final_score": final_score}

def prompt_time_value_balance_learner(total_hours, value_score, final_score):
    learner_prompt = f"""
    You are simulating a learner’s perception of whether the course was worth the time invested. The **Time-Value Balance** metric reflects how efficiently the course delivers educational value relative to its duration.

    **Evaluation Methodology:**
    - **Total Hours:** Total time required to complete the course.
    - **Value Score:** Learner-perceived value of the course (1 to 5), derived from course materials, structure, and alignment with goals.
    - **Value per Hour:** Value Score ÷ Total Hours — represents efficiency of learning.
    - **Final Rating:** Synthesizes all the above into a 1–5 score (5 = high value for time, 1 = poor return).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Whether the time spent on the course felt justified by the value it delivered.

    **Prompt Instruction:**
    Given Time Required: {total_hours}, Content Value: {value_score}, and Final Score: {final_score}, write a short learner-style reflection on the time-value experience. Avoid technical terms. Reflect how efficient and worthwhile the learning felt overall.
    
    Note:
    • Use simple, non-technical language that anyone can understand.
    • Do not mention or reference any of the provided metric names or values—these are confidential.
    • Focus on what the data implies or suggests, not on the metric itself.
    """
    return learner_prompt

def prompt_time_value_balance_instructor(total_hours, value_score, final_score):
    instructor_prompt = f"""
    You are an AI assistant reviewing educational content based on how well the course justifies the time it takes to complete. Feedback should be framed as a short, clear statement pointing out whether the course offers strong or weak value for the time invested.

    **What to do:**
    - Write 2-4 concise lines that sound like constructive feedback or an observation.
    - Do not explain how the evaluation was done or refer to any scoring system.
    - Focus only on what the course should improve or maintain based on how well the learning experience matches the time commitment.

    **Style and Tone:**
    - Use non-technical, everyday language.
    - Avoid mentioning any specific metric names or values.
    - Make the message sound like a direct observation or suggestion about the course's balance of content vs. duration.

    **Example formats:**
    - “Time is overestimated based on the value provided—consider improving content quality or tightening the scope.”
    - “The course offers strong value for the time invested—keep up the focused delivery.”
    - “Consider simplifying or trimming down the course to better match the learning value.”
    - “Good pacing and clarity throughout—time is well justified by the content.”

    **Inputs:**
    - Total Duration: {total_hours} hours  
    - Assigned Value: {value_score}  
    - Final Evaluation Score: {final_score}
    """
    return instructor_prompt

def learner_expectation_course_outcome_analysis(service_account_file, folder_id):
    result = {}
    course_name = get_folder_name(folder_id, service_account_file)
    result["Course Name"] = course_name
    service = get_gdrive_service(service_account_file)
    # Try to fetch metadata.json
    metadata = fetch_metadata_json(service, folder_id)
    course_info = metadata["About this Course"]
    difficulty_level = metadata["Level"]
    prerequisites = metadata["Prerequisites"]["Technical Prerequisites"] + metadata["Prerequisites"]["Conceptual Prerequisites"]
    commitment = metadata["Commitment"]
    learning_objectives = []
    syllabus = []
    for key, value in metadata.items():
        if key.startswith("Module") and isinstance(value, dict):
            module_los = value.get("Learning Objectives", [])
            learning_objectives.extend(module_los)
            module_syllabus = value.get("syllabus", [])
            syllabus.append(module_syllabus)
    result["Course Level Evaluation"] = {}
    tvb_score = get_tvb_score(course_info, difficulty_level, prerequisites, commitment, learning_objectives, syllabus)
    result["Course Level Evaluation"]["Time Value Balance Score"] = tvb_score["final_score"]
    result["Course Level Evaluation"]["Learner Perspective Assessment"] = execute_prompt(prompt_time_value_balance_learner(tvb_score["total_hours"], tvb_score["value_score"], tvb_score["final_score"]))
    instructor_feedback = execute_prompt(prompt_time_value_balance_instructor(tvb_score["total_hours"], tvb_score["value_score"], tvb_score["final_score"]))
    result["Course Level Evaluation"]["Instructor Feedback"] = re.sub("\n", "", instructor_feedback)
    save_dict_to_json(result, "Learner Expectation & Outcome Alignment - Time Value Balance Result.py")

if __name__ == "__main__":
    service_account_file = "slack-project.json"
    folder_id = "157Wq783qdGqBUoFbpYTsMKAQRPzi_5WX"
    start_time = time.time()
    learner_expectation_course_outcome_analysis(service_account_file, folder_id)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Time taken: {elapsed_minutes:.2f} minutes")
