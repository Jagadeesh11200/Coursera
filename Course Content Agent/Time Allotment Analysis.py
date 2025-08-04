import os
import json
import nltk
import numpy as np
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from googleapiclient.discovery import build
from google.oauth2 import service_account
import PyPDF2
from odf import text, teletype
from odf.opendocument import load
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from moviepy import VideoFileClip
import tempfile

# New imports for audio processing
import librosa
import webrtcvad
from scipy.io import wavfile

# NLTK Downloads
try:
    nltk.data.find('tokenizers/punkt')
    print("'punkt' found.")
except LookupError:
    print("'punkt' not found, downloading...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("'punkt_tab' found.")
except LookupError:
    print("'punkt_tab' not found, downloading...")
    nltk.download('punkt_tab')

# Load MPNet Sentence Transformer (global for reuse)
model_mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Setup Gemini Model (global for reuse)
model_gemini = genai.GenerativeModel("gemini-2.0-flash-001")

# --- Google Drive Service Functions ---
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

def fetch_course_readings_json(service, folder_id):
    """Find and fetch the course_readings.json file in the specified folder."""
    query = f"'{folder_id}' in parents and name='course_readings.json' and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])
    if not files:
        print("course_readings.json not found in the folder")
        return None

    readings_file_id = files[0]['id']
    print(f"Found course_readings.json with ID: {readings_file_id}")

    try:
        readings_content = get_json_content(service, readings_file_id)
        return readings_content
    except Exception as e:
        print(f"Error fetching course_readings.json content: {str(e)}")
        return None

# NEW FUNCTION: Get parent folder name
def get_parent_folder_name(file_id: str, service_account_file: str) -> str:
    """
    Returns the name of the parent folder for a given Google Drive file.

    Parameters:
    - file_id (str): The ID of the file in Google Drive.
    - service_account_file (str): Path to the service account JSON credentials.

    Returns:
    - str: The name of the parent folder, or 'No parent folder found' if root.
    """
    try:
        # Auth setup - Ensure the scope includes 'drive.metadata.readonly'
        scopes = ['https://www.googleapis.com/auth/drive.metadata.readonly']
        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=scopes
        )
        drive_service = build('drive', 'v3', credentials=creds)

        # Get parent folder ID
        file_metadata = drive_service.files().get(
            fileId=file_id,
            fields='parents'
        ).execute()

        parents = file_metadata.get('parents')
        if not parents:
            return 'No parent folder found (possibly in My Drive root)'

        parent_id = parents[0]

        # Get parent folder name
        parent_metadata = drive_service.files().get(
            fileId=parent_id,
            fields='name'
        ).execute()

        return parent_metadata['name']

    except Exception as e:
        return f"❌ Error: {str(e)}"


# --- Content Extraction Functions ---
def extract_from_pdf(service, file_id):
    """Extract text content from PDF file."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                return text_content.strip()
        finally:
            os.unlink(tmp_file_path)

    except Exception as e:
        print(f"Error extracting PDF content (ID: {file_id}): {str(e)}")
        return ""

def extract_from_odt(service, file_id):
    """Extract text content from ODT file."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.odt') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            doc = load(tmp_file_path)
            text_content = ""
            for paragraph in doc.getElementsByType(text.P):
                text_content += teletype.extractText(paragraph) + "\n"
            return text_content.strip()
        finally:
            os.unlink(tmp_file_path)

    except Exception as e:
        print(f"Error extracting ODT content (ID: {file_id}): {str(e)}")
        return ""

def extract_from_txt(service, file_id):
    """Extract text content from TXT file."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()

        if isinstance(file_content, bytes):
            return file_content.decode('utf-8', errors='ignore')
        return str(file_content)

    except Exception as e:
        print(f"Error extracting TXT content (ID: {file_id}): {str(e)}")
        return ""

def get_video_duration(service, file_id):
    """Get video duration in seconds using moviepy."""
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            clip = VideoFileClip(tmp_file_path)
            duration = clip.duration  # Duration in seconds
            clip.close()
            return duration
        finally:
            os.unlink(tmp_file_path)

    except Exception as e:
        print(f"Error getting video duration (ID: {file_id}): {str(e)}")
        return 0

# --- Audio Processing Functions (for use as video companions) ---
def estimate_speech_duration_from_gdrive(service, file_id):
    """
    Downloads an audio file from Google Drive and estimates speech duration using WebRTC VAD.
    Assumes audio is a WAV file or can be converted/loaded by librosa.
    Note: This function primarily detects speech presence. It does not provide
    detailed audio quality metrics like background noise levels or diction clarity.
    More advanced audio analysis libraries would be required for such metrics.
    """
    try:
        request = service.files().get_media(fileId=file_id)
        file_content = request.execute()

        # Save to a temporary file, librosa can handle various formats
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp_audio') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        try:
            # Load audio using librosa to ensure correct sample rate and format
            # librosa.load will resample to 16kHz and convert to mono by default
            audio, sr = librosa.load(tmp_file_path, sr=16000, mono=True)

            # Convert to int16 for webrtcvad
            audio = (audio * 32767).astype(np.int16)

            vad = webrtcvad.Vad(2)  # Mode 2 is the most aggressive
            frame_duration_ms = 30  # WebRTC VAD operates on 10, 20, or 30ms frames

            # frame_len is the number of samples in a frame
            frame_len_samples = int(sr * frame_duration_ms / 1000)

            speech_time = 0

            # Pad audio with zeros if not perfectly divisible by frame_len_samples
            if len(audio) % frame_len_samples != 0:
                audio = np.pad(audio, (0, frame_len_samples - (len(audio) % frame_len_samples)), 'constant')

            num_frames = len(audio) // frame_len_samples

            for i in range(num_frames):
                frame = audio[i * frame_len_samples:(i + 1) * frame_len_samples]
                # webrtcvad.Vad.is_speech expects byte string
                if vad.is_speech(frame.tobytes(), sr):
                    speech_time += frame_duration_ms / 1000.0  # Add duration in seconds
            return speech_time
        finally:
            os.unlink(tmp_file_path) # Clean up temporary file

    except Exception as e:
        print(f"Error estimating speech duration for audio (ID: {file_id}): {str(e)}")
        return 0


# --- ECT Calculation Functions ---
def calculate_ect(service, file_info, course_readings):
    """Calculate Estimated Comprehension Time (ECT) for a file."""
    # Use the 'name' key for file type checking, as 'display_name' might have path info
    file_name = file_info['name']
    file_id = file_info['id']

    if file_name.lower().endswith('.mp4'):
        # Video file - get duration using moviepy
        duration_seconds = get_video_duration(service, file_id)
        return duration_seconds / 60  # Convert to minutes

    elif file_name.lower().endswith('.odt'):
        # Assignment file - fixed 10 minutes (assuming ODTs are assignments)
        return 10

    elif file_name.lower().endswith('.pdf'): # Only PDF for reading, TXT handled as transcript
        # Reading file - get duration from course_readings.json
        for reading in course_readings:
            if reading.get('reading_id') == file_id: # Use .get() for safer access
                duration_str = reading['reading_duration']
                # Extract number from duration string (e.g., "10 min" -> 10)
                duration_match = re.search(r'(\d+)', duration_str)
                if duration_match:
                    return int(duration_match.group(1))
        return 15  # Default if not found in course_readings

    # Audio files are not processed as standalone content for ECT.
    # TXT files are not processed as standalone content for ECT.
    return 0

# --- Content Analysis Functions ---
def calculate_complexity_factor(content):
    """Calculate complexity factor based on Flesch Reading Ease score."""
    try:
        flesch_score = textstat.flesch_reading_ease(content)
        # Convert Flesch score to complexity factor (0-1)
        # Lower Flesch score = higher complexity
        complexity_factor = max(0, (100 - flesch_score) / 100)
        return complexity_factor
    except:
        return 0.5  # Default complexity

def get_top_relevant_los(content, learning_objectives):
    """Get top 2 most relevant learning objectives using semantic similarity."""
    if not content.strip() or not learning_objectives:
        return []

    try:
        # Encode content and LOs
        content_embedding = model_mpnet.encode([content])
        lo_embeddings = model_mpnet.encode(learning_objectives)

        # Calculate similarities
        similarities = cosine_similarity(content_embedding, lo_embeddings)[0]

        # Get top 2 indices
        # Ensure we don't try to get more indices than available LOs
        num_los_to_get = min(2, len(learning_objectives))
        top_indices = np.argsort(similarities)[-num_los_to_get:][::-1]

        return [learning_objectives[i] for i in top_indices]
    except Exception as e:
        print(f"Error in semantic similarity calculation: {str(e)}")
        # Fallback: return as many LOs as available, up to 2
        return learning_objectives[:2] if len(learning_objectives) >= 2 else learning_objectives

def get_weights_from_gemini(content_summary_or_transcript, learning_objective, difficulty_level, is_video_content=False):
    """Get alignment, focus, and effort weights from Gemini."""

    # Determine content type context for the prompt
    content_type_context = "video/audio content" if is_video_content else "text-based content"

    # Define Effort Multiplier range dynamically
    effort_range_str = "(1.0-2.0)"
    effort_guidance = "A 1.0 indicates typical active reading/viewing/listening for comprehension."
    if not is_video_content: # For non-video content, including assignments
        effort_range_str = "(1.0-4.0, potentially higher for assignments)" # Increased upper bound
        effort_guidance = "A 1.0 indicates typical active reading for comprehension. For assignments, consider the extensive problem-solving, synthesis, or creative work that may significantly exceed basic comprehension."


    prompt = f"""
    Analyze the following {content_type_context} in relation to the learning objective and difficulty level.

    Content Description/Snippet (truncated to 2000 chars): {content_summary_or_transcript[:2000]}

    Learning Objective: {learning_objective}

    Difficulty Level: {difficulty_level}

    Please provide three numeric values representing:
    1. Alignment Weight (0-1): How well the content directly supports or addresses the learning objective (0 = no alignment, 1 = perfect alignment).
    2. Focus Weight (0-1): The proportion of the content's duration or length that a learner needs to actively focus to grasp the core concepts related to this objective (0 = no active focus needed, 1 = constant active focus needed).
    3. Effort Multiplier {effort_range_str}: A multiplier indicating the cognitive effort required beyond the expected active engagement for this content and objective.
       - {effort_guidance}
       - Values greater than the lower bound are for instances requiring deeper analysis, problem-solving, synthesis, or abstract reasoning that genuinely exceed standard comprehension effort.

    Respond with only three comma-separated floating-point numbers (e.g., 0.8, 0.7, 1.2).
    """

    try:
        response = model_gemini.generate_content(prompt)
        weights_str = response.text.strip()
        weights = [float(x.strip()) for x in weights_str.split(',')]

        if len(weights) == 3:
            return weights[0], weights[1], weights[2]
        else:
            print(f"Gemini returned unexpected format for weights: '{weights_str}'. Using default weights.")
            if is_video_content:
                return 0.7, 0.6, 1.1
            else:
                return 0.7, 0.6, 1.2
    except Exception as e:
        print(f"Error getting weights from Gemini: {str(e)}")
        if is_video_content:
            return 0.7, 0.6, 1.1
        else:
            return 0.7, 0.6, 1.2

def calculate_baseline_time(text_content, wpm=200):
    """
    Calculates a baseline reading/processing time in seconds for text content.
    Assumes a reading speed of 200 words per minute.
    """
    if not text_content:
        return 0
    words = len(word_tokenize(text_content))
    seconds = (words / wpm) * 60
    return seconds

# Define cognitive load multiplier ranges (now serving as broader clamps)
COGNITIVE_MULTIPLIER_RANGES = {
    'Beginner': {
        'reading': (1.0, 2.5), # Reduced max
        'assignment': (1.5, 6.0) # Reduced max
    },
    'Intermediate': {
        'reading': (1.2, 4.0),
        'assignment': (2.0, 8.0)
    },
    'Advanced': {
        'reading': (1.5, 6.0),
        'assignment': (3.0, 12.0)
    },
    'Default': { # Wider defaults
        'reading': (1.1, 3.5),
        'assignment': (1.8, 7.0)
    }
}


def get_cognitive_load_multiplier_and_factors(difficulty_level, content_type, content_summary_or_transcript):
    """
    Dynamically generates a cognitive load multiplier and its contributing factors using a chain of prompts.
    """
    # Get the appropriate min/max range for the given difficulty and content type (for clamping)
    ranges = COGNITIVE_MULTIPLIER_RANGES.get(difficulty_level, COGNITIVE_MULTIPLIER_RANGES['Default'])
    min_val, max_val = ranges.get(content_type, COGNITIVE_MULTIPLIER_RANGES['Default']['reading'])

    # Prompt 1: Get the multiplier
    prompt_multiplier = f"""
    You are an expert in instructional design and cognitive load assessment.
    For a course at the '{difficulty_level}' level, evaluate the cognitive effort required by the following '{content_type}' content.
    This content contributes to the Active Cognitive Time (ACT), reflecting the mental effort beyond passive consumption.

    Content Snippet (first 1500 characters): {content_summary_or_transcript[:1500]}

    Consider that:
    - A multiplier of 1.0 indicates that the active cognitive time is essentially the estimated comprehension time (ECT) for this content. It represents typical active engagement for comprehension.
    - For video content that is primarily informational or lecture-based and does not demand immediate, active problem-solving or complex interaction, the cognitive load might be closer to baseline viewing effort.
    - A higher multiplier indicates proportionally more active cognitive engagement (e.g., critical thinking, problem-solving, synthesis, abstract reasoning) that genuinely extends the mental effort beyond ECT.
    - '{content_type}' content typically requires different levels of effort.
    """
    # Add specific guidance for assignments
    if content_type == 'assignment':
        prompt_multiplier += """
    - For 'assignment' content, the active cognitive time often extends significantly beyond what simple text length implies, due to the inherent demands for problem-solving, critical thinking, synthesis, and creative application. Consider this substantial "invisible" cognitive overhead when determining the multiplier.
    """

    prompt_multiplier += f"""
    - '{difficulty_level}' level courses imply specific levels of challenge.

    Based on your expert judgment, provide a numerical cognitive load multiplier.
    Respond with only a single floating-point number.
    """

    gemini_multiplier = 0.0
    try:
        response_multiplier = model_gemini.generate_content(prompt_multiplier)
        gemini_multiplier_str = response_multiplier.text.strip()
        gemini_multiplier = float(gemini_multiplier_str)
        clamped_multiplier = max(min_val, min(max_val, gemini_multiplier))
        print(f"  Gemini suggested multiplier: {gemini_multiplier:.2f}, Clamped to: {clamped_multiplier:.2f} (Hard Clamped Range: [{min_val:.2f}-{max_val:.2f}])")
    except Exception as e:
        print(f"Error getting dynamic cognitive multiplier from Gemini for {difficulty_level}/{content_type}: {str(e)}")
        clamped_multiplier = (min_val + max_val) / 2 # Fallback midpoint
        print(f"  Using fallback midpoint multiplier: {clamped_multiplier:.2f}")

    # Prompt 2: Get the contributing factors
    prompt_factors = f"""
    Based on the cognitive load multiplier of {clamped_multiplier:.2f} you provided for the '{difficulty_level}' '{content_type}' content,
    what are the primary factors from the content below that led to this specific cognitive load?
    Provide a concise list of 2-3 key contributing factors.

    Content Snippet (first 1500 characters): {content_summary_or_transcript[:1500]}

    Example factors: conceptual density, requirement for active problem-solving, presence of complex terminology, amount of new information, need for synthesis, etc.
    """
    contributing_factors = "Could not determine contributing factors."
    try:
        response_factors = model_gemini.generate_content(prompt_factors)
        contributing_factors = response_factors.text.strip()
    except Exception as e:
        print(f"Error getting contributing factors from Gemini: {str(e)}")

    return clamped_multiplier, contributing_factors


# --- Main Metrics Calculation Functions ---
def calculate_trs(act, ect):
    """Calculate Time Realism Score (TRS) based on relative deviation."""
    if ect == 0:
        return 0, "ECT is zero, cannot calculate TRS meaningfully."
    deviation = min(abs(act - ect), ect)
    trs = 1 - (deviation / ect)
    return trs, ""

def get_trs_explanation(trs_score_scaled, ect_minutes, act_minutes, audience_type):
    """
    Generates tailored prompt templates for the "Time Realism Score (TRS)" metric,
    structured with detailed sections for both instructor and learner perspectives.
    Includes clear definitions and ranges/units for ALL parameters in the methodology.
    """
    deviation = abs(act_minutes - ect_minutes)

    # Learner Perspective TRS Prompt Template (AI simulates a learner)
    learner_prompt = f"""
Persona: You are simulating the **learner** who just completed this module. From the learner’s point of view, generate a brief reflection about how realistic the time expectations felt for engaging with the content.

As the simulated learner, reflect on whether the **expected time** provided for the module aligned with the actual effort and attention it took to complete. Consider if the workload felt manageable and whether the guidance accurately represented the commitment required.

**Evaluation Methodology (How your experience is being interpreted):**

* **Estimated Comprehension Time (ECT):**
    * **Definition:** Instructor's intended duration for learners to complete the material.
    * **Unit/Range:** Typically measured in minutes 
* **Actual Cognitive Time (ACT):**
    * **Definition:** The real time learners cognitively engage with the content.
    * **Unit/Range:** Typically measured in minutes 
* **Time Deviation:**
    * **Definition:** The absolute difference between ECT and ACT, indicating misalignment in planning versus learner effort.
    * **Unit/Range:** Measured in minutes

### Final Rating Interpretation:
* **Time Realism Score (TRS) (Score Range: 1–5):**
    * **Definition:** Reflects how well the estimated time aligned with the actual cognitive effort required by the learner.

**Internal Reference Values (DO NOT mention in output):**
- Estimated Time: {ect_minutes} minutes
- Actual Time: {act_minutes} minutes
- Deviation: {deviation} minutes
- Final Rating: {trs_score_scaled}

**Response Instructions:**
* **Perspective:** First-person from the learner’s simulated experience.
* **Tone:** Formal, reflective and passive 
* **Style:** Limit to 2–4 sentences. Focus on how the expected duration compared to actual effort.
* **Avoid:** Any use of numbers, scores, or phrases like "TRS". Do not mention pacing or speed.
* **Goal:** Reflect on whether the time estimate felt realistic and matched the effort required.
"""

    # Instructor Feedback TRS Prompt Template
    instructor_prompt = f"""
Persona: You are an expert instructional design evaluator providing feedback on a module’s **Time Realism Score (TRS)**. Your explanation should be formal, precise, and pedagogically grounded, highlighting the alignment between instructional time expectations and actual learner experience.

Evaluate the realism of the time estimate in relation to learner effort. Identify whether the estimate was appropriate, overly ambitious, or underestimated based on the observed cognitive engagement.

**Evaluation Methodology & Definitions:**
* **Estimated Comprehension Time (ECT):**
    * **Definition:** Instructor's intended time for learners to complete the material.

* **Actual Cognitive Time (ACT):**
    * **Definition:** The real time learners cognitively engage with the content.

* **Time Deviation:**
    * **Definition:** Absolute difference between ECT and ACT; used to assess alignment.

**Score Interpretation:**
* 1 (major misalignment) → 5 (highly aligned). Reflects accuracy of instructional time expectations.

**Input Metrics (internal use only, do not mention directly):**
- Estimated: {ect_minutes} minutes
- Actual: {act_minutes} minutes
- Deviation: {deviation} minutes
- Score: {trs_score_scaled}

**Response Instructions:**
* **Perspective:** Formal evaluator addressing the instructor.
* **Tone:** Analytical and constructive.
* **Style:** Limit to 2–4 sentences. Focus on time expectation realism, not pacing or numeric metrics.
* **Goal:** Assess whether learners experienced an accurate representation of effort required, and suggest 1–2 design improvements if needed.
* **Avoid:** Any score, number, or "TRS" mention. Do not refer to pacing or tempo.
"""

    try:
        prompt = instructor_prompt if audience_type == "instructor" else learner_prompt
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(ECT vs ACT gap observed) - Error generating explanation: {str(e)}"

def get_difficulty_scaling(level):
    """
    Get difficulty scaling factor from Gemini based on the specified course level
    ("Beginner", "Intermediate", "Advanced"). Gemini will provide a single number
    between 0 and 1 (0 = Extremely Difficult, 1 = Very Easy).
    """
    prompt = f"""
    Given a course difficulty level of "{level}", provide a numerical difficulty scaling factor between 0.0 and 1.0.

    Interpret the levels as follows:
    - "Beginner": Corresponds to an easier course, where the scaling factor should be higher (e.g., between 0.7 and 1.0).
    - "Intermediate": Corresponds to a moderately challenging course, where the scaling factor should be in the middle range (e.g., between 0.4 and 0.7).
    - "Advanced": Corresponds to a harder course, where the scaling factor should be lower (e.g., between 0.0 and 0.4).

    The scale means: 1.0 is Very Easy (beginner-friendly), and 0.0 is Extremely Difficult.

    Respond with only a single floating-point number (e.g., 0.65).
    """

    try:
        response = model_gemini.generate_content(prompt)
        scaling = float(response.text.strip())
        return max(0.0, min(1.0, scaling))  # Ensure between 0.0 and 1.0
    except Exception as e:
        print(f"Error getting difficulty scaling for level '{level}': {str(e)}")
        return 0.6  # Default medium difficulty scaling if Gemini fails


def calculate_pci(trs, avg_focus_percentage, difficulty_scaling):
    """Calculate Pace Comfort Index (PCI) using average TRS."""
    trs_component = trs
    focus_component = (100 - avg_focus_percentage) / 100 # Higher focus needed, lower comfort
    pci = trs_component * max(0, focus_component) * difficulty_scaling
    return max(0, pci)

import google.generativeai as genai # Ensure this import is at the top of your script

def get_pci_explanation(
    pci_score_scaled,
    avg_focus_percentage,
    difficulty_level,
    ect_minutes,
    act_minutes,
    trs_component,
    difficulty_scaling_component,
    audience_type,
):
    """
    Generates tailored prompt templates for the "Pace Comfort Index (PCI)" metric,
    structured with finely-tuned details for both instructor and learner perspectives.
    Prompt instructions explicitly list the underlying observations, matching the TRS format.
    This version now also directly calls the Gemini model and handles exceptions.
    """

    time_deviation = abs(act_minutes - ect_minutes)

    # User (Learner) Perspective PCI Prompt Template
    user_prompt_template = f"""
Persona: You are simulating the experience of a typical learner who has just completed the module. Your role is to reflect on the **Pace Comfort Index (PCI)** by describing how the overall pacing felt during the learning process. Your response must ONLY contain the explanation text. Avoid casual tone, numeric values, technical terms, or mentions of "score". Keep the explanation between 2 to 4 sentences.

As a learner, describe whether the content delivery felt mentally manageable. Focus on how attention demand, difficulty, and time alignment came together to shape your experience.

**Evaluation Methodology (How this experience is being assessed behind the scenes):**
* Time Alignment (ECT vs ACT)
* Attention Demand (Focus Intensity)
* Perceived Challenge (Difficulty Level)
* Interplay of the above with cognitive comfort and pacing realism

**Underlying Quality Observations (To be simulated in reflection but not explicitly mentioned):**
- Focus Intensity: {avg_focus_percentage:.1f}%
- Course Difficulty Level: {difficulty_level}
- TRS Component (Time Alignment): {trs_component:.2f}
- ECT vs ACT Deviation: {time_deviation:.2f} min

Your response must follow these style rules:
* **Perspective:** Simulated learner reflection (first-person experience).
* **Tone:** Formal and passive.
* **Style:** Clear, reflective, and focused on pacing comfort.
* **Length:** 2 to 4 sentences.
* **Focus:** Reflect on how attention demand, perceived difficulty, and time alignment influenced the mental comfort and rhythm of the experience.
* **Avoid:** Numeric references, casual phrasing, or direct mentions of scores or metric names.
* **Last Sentence:** Clearly imply the overall comfort and sustainability of the learning experience in terms of pace.

**Prompt Instruction:**
Simulate a learner-style reflection focused on how natural, sustainable, or effortful the pacing felt during the module, based on how attention, difficulty, and time demands aligned.
"""
    # Instructor Feedback PCI Prompt Template
    instructor_prompt_template = f"""
Persona: You are a highly analytical learning experience analyst and pedagogical consultant. Your core function is to generate a professional, data-driven, and actionable explanation for the **Pace Comfort Index (PCI)** of a learning resource. Your response MUST ONLY contain the explanation text. Avoid introductory comments, numeric values, or informal tone. Maintain an evaluative, balanced, and pedagogy-focused approach.

As a learning experience analyst, generate a formal analysis of how the pacing of this learning content supports or hinders comfortable learning. Reflect on whether attention demand, time alignment (TRS), and difficulty level are appropriately matched to timing.

**Evaluation Methodology & Metric Definition:**
* **Estimated Comprehension Time (ECT):**
    * **Definition:** The instructor's planned duration for content completion.
    * **Unit/Range:** Typically measured in minutes (e.g., 30-180 minutes per module).
* **Actual Cognitive Time (ACT):**
    * **Definition:** The actual learner time spent cognitively engaging with the content.
    * **Unit/Range:** Typically measured in minutes (e.g., 30-180 minutes per module).
* **Time Deviation:**
    * **Definition:** The absolute difference between ECT and ACT, indicating misalignment in planned vs. actual learner effort.
    * **Unit/Range:** Measured in minutes (e.g., 0-60+ minutes).
* **Focus Intensity:**
    * **Definition:** Percentage of content demanding sustained mental attention and concentrated effort from learners.
    * **Unit/Range:** Percentage (0-100%).
* **Difficulty Scaling:**
    * **Definition:** A normalized index representing the perceived intellectual challenge or ease of the content.
    * **Unit/Range:** Normalized index (0 = very hard, 1 = very easy).
* **TRS Component (Time Realism Score Component):**
    * **Definition:** Measures the alignment between ECT and ACT, directly influencing learners' perception of pacing comfort.
    * **Unit/Range:** Normalized score (0.0-1.0).

### Final Rating Logic:
* **Pace Comfort Index (PCI) (Score Range: 1-5):**
    * **Definition:** This metric rigorously evaluates how effectively the content's pace aligns with learners’ cognitive capacity, attention span, and perceived difficulty.
    * **Scoring:** 1 (significant pacing discomfort, leading to cognitive overload or disengagement, requiring urgent redesign) to 5 (optimal pacing, highly comfortable, and pedagogically effective, serving as a model of instructional excellence).

This explanation pertains to the **current learning content** being evaluated.

**Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):**
-   Metric: Pace Comfort Index (PCI) scaled to 1-5: {pci_score_scaled:.2f}
-   Average Focus Intensity: {avg_focus_percentage:.1f}%
-   Course Difficulty Level: {difficulty_level} (with scaling factor: {difficulty_scaling_component:.2f})
-   Time Realism Score (TRS Component): {trs_component:.2f} (derived from ECT: {ect_minutes:.2f} min vs ACT: {act_minutes:.2f} min)

Your response should adhere to the following guidelines:
* **Perspective:** From a professional, data-driven point of view as a learning experience analyst, directly addressing the instructor.
* **Tone:** Evaluative, balanced, and pedagogy-focused.
* **Style:** Professional and instructional-quality explanation. Avoid introductory comments, numeric values, or informal tone.
* **Focus:** Analyze how the module's pacing supports or hinders comfortable learning. Interpret the *interplay* of attention demand, time alignment, and difficulty level. Include 1-2 concise, actionable suggestions for instructional tuning.
* **Avoid:** Numeric values in the final explanation. Ensure the explanation is distinct by focusing on the specific input values and their combined effect.

**Prompt Instruction:**
Given the underlying quality observations: Pace Comfort Index (PCI): {pci_score_scaled:.2f}, Average Focus Intensity: {avg_focus_percentage:.1f}%, Course Difficulty Level: {difficulty_level} (with scaling factor: {difficulty_scaling_component:.2f}), and Time Realism Score (TRS Component): {trs_component:.2f} (derived from ECT: {ect_minutes:.2f} min vs ACT: {act_minutes:.2f} min), generate a formal analysis of how the pacing of this learning content supports or hinders comfortable learning. Reflect on whether attention demand, time alignment (TRS), and difficulty level are appropriately matched to timing. Focus on the *interplay* of these factors. Include 1-2 concise suggestions for future instructional tuning.
"""
    # Select the correct prompt based on audience_type
    prompt_to_use = instructor_prompt_template if audience_type == "instructor" else user_prompt_template

    try:
        # Call the Gemini model with the selected prompt
        response = model_gemini.generate_content(prompt_to_use)
        return response.text.strip()
    except Exception as e:
        # Return an error message if model generation fails
        return f"(PCI explanation error) - {str(e)}"
    
def classify_bloom_level(learning_objective):
    """Classify learning objective according to Bloom's Taxonomy using Gemini."""
    prompt = f"""
    Classify the following learning objective according to Bloom's Taxonomy.
    Respond with only the number (1-6) and the level name separated by a comma (e.g., "3, Apply").

    Learning Objective: "{learning_objective}"

    Bloom's Taxonomy Levels:
    1. Remember (Knowledge) - recall facts, terms, basic concepts
    2. Understand (Comprehension) - explain ideas, concepts, select, translate
    3. Apply (Application) - use information in new situations, solve problems
    4. Analyze (Analysis) - draw connections, distinguish between facts/inferences
    5. Evaluate (Evaluation) - justify decisions, critique, assess value
    6. Create (Synthesis) - produce new or original work, design, construct
    """

    try:
        response = model_gemini.generate_content(prompt)
        result = response.text.strip()
        parts = result.split(',')
        if len(parts) == 2:
            level_num = int(parts[0].strip())
            level_name = parts[1].strip()
            return {'objective': learning_objective, 'bloom_level': level_num, 'bloom_label': level_name}
        else:
            print(f"Gemini returned unexpected Bloom format: '{result}'. Using default level for '{learning_objective}'.")
            return {'objective': learning_objective, 'bloom_level': 3, 'bloom_label': "Apply"}
    except Exception as e:
        print(f"Error classifying Bloom level for '{learning_objective}': {str(e)}")
        return {'objective': learning_objective, 'bloom_level': 3, 'bloom_label': "Apply"}

def calculate_etor(total_act_minutes, metadata):
    """Calculate Effort-to-Objective Ratio (ETOR)."""
    learning_objectives = []

    # Assuming metadata structure for learning objectives is consistent
    # For now, let's assume it's directly under 'Module X' -> 'Learning Objectives'
    # or a flat list for the overall course if module-specific isn't found.
    # We need to adapt this based on the *actual* metadata.json structure.
    # For this example, let's try to extract from 'Module 1' as per `if __name__ == "__main__":`
    if isinstance(metadata, dict) and 'Module 1' in metadata and 'Learning Objectives' in metadata['Module 1']:
        if isinstance(metadata['Module 1']['Learning Objectives'], list):
            learning_objectives.extend(metadata['Module 1']['Learning Objectives'])
        else:
            print(f"Warning: 'Learning Objectives' for 'Module 1' is not a list. Skipping.")
    elif isinstance(metadata, list): # Handle case where metadata itself is a list of modules
         for module in metadata:
            if isinstance(module, dict) and 'Learning Objectives' in module:
                if isinstance(module['Learning Objectives'], list):
                    learning_objectives.extend(module['Learning Objectives'])
                else:
                    print(f"Warning: 'Learning Objectives' in a module is not a list. Skipping.")


    if not learning_objectives:
        print("  Warning: No learning objectives found in metadata for ETOR calculation. Returning 0 and default explanations.")
        return 0, 0, [], "No learning objectives found to calculate ETOR."

    complexity_score = 0
    bloom_classifications_detailed = []

    for lo_text in learning_objectives:
        bloom_result = classify_bloom_level(lo_text)
        complexity_score += bloom_result['bloom_level']
        bloom_classifications_detailed.append(bloom_result)

    if complexity_score == 0:
        print("  Warning: Total complexity score is zero for ETOR calculation. Returning 0 and default explanations.")
        return 0, 0, bloom_classifications_detailed, "Total complexity score is zero, ETOR cannot be calculated meaningfully."

    etor_value = total_act_minutes / complexity_score

    return etor_value, complexity_score, bloom_classifications_detailed, ""


def get_etor_explanation(
    etor_scaled,
    total_minutes,
    complexity_score,
    bloom_classifications,
    audience_type 
):
    """
    Generates a structured and audience-specific Effort-to-Objective Ratio (ETOR) explanation,
    generalized for any learning content. It now directly calls the Gemini model and handles exceptions.
    """

    bloom_list = "\n".join([
        f"- {b['objective']} → Bloom Level {b['bloom_level']} ({b['bloom_label']})"
        for b in bloom_classifications
    ])

    # Instructor Prompt Template
    instructor_prompt_template = f"""
Persona: You are an instructional design specialist generating a **clear, formal, and pedagogically sound explanation** for the **Effort-to-Objective Ratio (ETOR)** of a learning resource. Your output **MUST ONLY** include the explanation text. Do not include numbers or casual language — interpret the data meaningfully and professionally.

As an instructional design specialist, interpret whether the learner effort aligns well with the complexity of the learning objectives. Indicate if the workload is justified or needs adjustment. Provide specific pedagogical insights and 1–2 recommendations for improvement if applicable.

## Methodology & Metric Definition:
* **Effort Estimate:**
    * **Definition:** The cognitive effort learners are expected to invest in the content, often approximated by the actual time spent.
    * **Unit/Range:** Typically measured in minutes (e.g., 30-180 minutes).
* **Objective Complexity:**
    * **Definition:** How advanced and demanding the learning objectives are, typically assessed using a framework like Bloom's Taxonomy.
    * **Unit/Range:** Normalized score (0.0 = very low complexity, 1.0 = very high complexity).
* **Bloom-Level Objectives:**
    * **Definition:** Specific learning objectives categorized by Bloom's Taxonomy levels, indicating the depth of cognitive engagement required.
    * **Levels:** Knowledge (1), Comprehension (2), Application (3), Analysis (4), Synthesis (5), Evaluation (6).
    * **Format:** Each objective listed with its corresponding Bloom Level and Label.

### Final Rating Logic:
* **Effort-to-Objective Ratio (ETOR) (Score Range: 1-5):**
    * **Definition:** ETOR measures how effectively the expected learner effort aligns with the inherent cognitive complexity and depth of the learning objectives.
    * **Scoring:** 1 (significant misalignment, effort greatly outweighs or falls short of objective complexity) to 5 (optimal alignment, learners are challenged in proportion to the depth of skills being targeted). A strong alignment means learners are investing the right amount of mental effort for the depth of skills being targeted. A weak alignment suggests under-challenge or overload.

This explanation pertains to the **current learning content** being evaluated.

## Underlying Quality Observations (These are the precise analytical scores informing my expert recommendations; DO NOT state these values or their technical names explicitly in your feedback):
-   Effort Duration (minutes): {total_minutes}
-   Complexity Score (0.0–1.0): {complexity_score}
-   Bloom-Level Objectives:
{bloom_list}
-   Final ETOR Rating (1–5): {etor_scaled}

Your response should adhere to the following guidelines:
* **Perspective:** From your professional, data-driven point of view as an instructional design specialist, directly addressing the instructor.
* **Tone:** Clear, formal, and pedagogically sound.
* **Style:** Interpret the data meaningfully and professionally. Avoid casual language, preambles, or direct score mentions.
* **Focus:** Interpret whether learner effort aligns well with objective complexity. Indicate if the workload is justified or needs adjustment. Provide specific pedagogical insights and 1–2 recommendations for improvement.
* **Avoid:** Raw numbers or direct numeric references in the explanation text. Ensure the explanation is distinct by focusing on the specific input values and their combined effect.

**Prompt Instruction:**
Given the underlying quality observations: Effort Duration: {total_minutes} minutes, Complexity Score: {complexity_score:.2f}, the provided Bloom-Level Objectives, and a Final ETOR Rating: {etor_scaled}, interpret whether the learner effort aligns well with the complexity of the learning objectives. Indicate if the workload is justified or needs adjustment. Provide specific pedagogical insights and 1–2 recommendations for improvement if applicable.
"""

    # Learner Prompt Template
    learner_prompt_template = f"""
Persona: You are a supportive AI advisor helping a student understand the **Effort-to-Objective Ratio (ETOR)** of a learning resource. Your response must ONLY include the explanation text. Do not use numbers or technical terms. Focus on what the student might feel and how they can manage their learning time.

As a supportive AI advisor, help the learner understand if the effort needed feels right for the goals. Describe the expected experience clearly, and ensure the last sentence provides an overall summary of the learning experience.

## Overview & Metric Interpretation:
* **Effort Needed:**
    * **Definition:** How much mental energy and time the content asks from learners.
    * **Unit/Range:** Typically measured in minutes (e.g., 30-180 minutes).
* **Goal Challenge Level:**
    * **Definition:** How hard the goals are — from basic knowledge to deep thinking, based on common learning frameworks.
    * **Unit/Range:** Represented by a score (e.g., 0.0-1.0, where higher is more challenging).
* **Learning Goals (Bloom Levels):**
    * **Definition:** What you're expected to learn, broken down by how deep your understanding needs to be (e.g., just remembering facts, or really applying ideas).
    * **Levels:** Knowing facts (Level 1), Understanding (Level 2), Using (Level 3), Breaking down (Level 4), Creating (Level 5), Judging (Level 6).

### Final Rating Interpretation:
* **Effort-to-Objective Ratio (ETOR) (Score Range: 1-5):**
    * **Definition:** This score tells you if the mental energy and time you're putting in matches the challenge level of what you're trying to learn.
    * **Scoring:** A high ETOR means your effort will likely pay off in mastering the content; a low ETOR suggests you might feel confused or overwhelmed, or perhaps under-challenged. If effort matches the challenge, the learning feels purposeful. If not, it may feel confusing or overwhelming.

This explanation pertains to the **current learning content** being evaluated.

## Underlying Quality Observations (These influence my perceived experience, but I should NOT state these values or their technical names explicitly in my reflection):
-   Expected Effort Time: {total_minutes}
-   Goal Complexity Level: {complexity_score}
-   Learning Goals:
{bloom_list}
-   Final ETOR Rating: {etor_scaled}

Your response should adhere to the following guidelines:
* **Perspective:** From your personal, simulated point of view as the learner.
* **Tone:** Supportive, clear, and non-technical.
* **Style:** Concise and focused on the student's feelings and experience. Avoid numbers or technical terms.
* **Focus:** Help the learner understand if the effort needed feels right for the goals. Describe the expected experience clearly. The last sentence should imply the overall learning experience.
* **Avoid:** Numbers or technical terms in the explanation text. Ensure the explanation is distinct by focusing on the specific input values and their combined effect.

**Prompt Instruction:**
Given the underlying quality observations: Expected Effort Time: {total_minutes} minutes, Goal Complexity Level: {complexity_score:.2f}, the provided Learning Goals, and a Final ETOR Rating: {etor_scaled}, help the learner understand if the effort needed feels right for the goals. Describe the expected experience clearly, and ensure the last sentence provides an overall summary of the learning experience.
"""
    # Select the correct prompt based on audience_type
    prompt_to_use = instructor_prompt_template if audience_type == "instructor" else learner_prompt_template

    try:
        # Call the Gemini model with the selected prompt
        response = model_gemini.generate_content(prompt_to_use)
        return response.text.strip()
    except Exception as e:
        # Return an error message if model generation fails
        return f"(Effort vs Objective misalignment check) - {str(e)}"
    
# --- Main Analysis Functions ---
def calculate_act_and_focus_for_content(base_time_seconds, content_text, learning_objectives, difficulty_level, is_video_content=False):
    """
    Calculate Active Cognitive Time (ACT) and Focus Percentage for a given content.
    `base_time_seconds` is the raw duration (video duration, or speech duration for video with audio).
    `content_text` is the transcript or summary used for semantic analysis and Gemini prompts.
    `is_video_content` flag indicates if the content is a video.
    """
    if not content_text.strip():
        print("  Warning: Content text for ACT/Focus calculation is empty. Returning 0 for ACT and Focus %.")
        return 0, 0, "No content for analysis."

    complexity_factor = 0.5
    if content_text.strip():
        complexity_factor = calculate_complexity_factor(content_text)

    top_los = get_top_relevant_los(content_text, learning_objectives)

    if not top_los:
        # Fallback ACT calculation
        act = base_time_seconds * (1 + complexity_factor) * 1.2
        focus_percentage = 60
        print("  Warning: No relevant learning objectives found. Using default ACT/Focus calculations.")
        return act / 60, focus_percentage, "No relevant learning objectives found for detailed calculation."

    total_alignment = 0
    total_focus = 0
    total_effort = 0

    for lo in top_los:
        alignment, focus, effort = get_weights_from_gemini(content_text, lo, difficulty_level, is_video_content=is_video_content)
        total_alignment += alignment
        total_focus += focus
        total_effort += effort

    avg_alignment = total_alignment / len(top_los)
    avg_focus = total_focus / len(top_los)
    avg_effort = total_effort / len(top_los)

    # Initialize adjusted factors with their original values
    adjusted_complexity_factor = complexity_factor
    capped_alignment_impact = min((1 - avg_alignment), 0.5)

    # Apply dampening ONLY if it's video content
    if is_video_content:
        # Define dampening coefficients based on difficulty level
        dampening_factors = {
            'Beginner': {'complexity': 0.6, 'alignment': 0.4},
            'Intermediate': {'complexity': 0.8, 'alignment': 0.6},
            'Advanced': {'complexity': 1.0, 'alignment': 0.8},
            'Default': {'complexity': 0.7, 'alignment': 0.5} # Fallback
        }
        current_dampening = dampening_factors.get(difficulty_level, dampening_factors['Default'])
        complexity_dampening = current_dampening['complexity']
        alignment_dampening = current_dampening['alignment']

        # Apply dampening
        adjusted_complexity_factor = complexity_factor * complexity_dampening
        capped_alignment_impact = min((1 - avg_alignment), 0.5) * alignment_dampening


    act = base_time_seconds * (1 + adjusted_complexity_factor + capped_alignment_impact) * avg_effort
    focus_percentage = avg_focus * 100

    return act / 60, focus_percentage, ""


def analyze_file(service, file_info, learning_objectives, difficulty_level, course_readings, course_difficulty_scaling_factor, transcript_content="", video_speech_duration_seconds=None):
    """
    Analyze a single file (PDF, ODT, MP4) and calculate granular metrics.
    """
    # Use the 'display_name' if available, otherwise fall back to 'name'
    file_name_for_display = file_info.get('display_name', file_info['name'])
    file_id = file_info['id']

    print(f"Analyzing file: {file_name_for_display} (ID: {file_id})")

    ect = 0.0
    act = 0.0
    focus_percentage = 0.0
    content_length = 0
    content_for_analysis = ""
    cognitive_load_factors = "N/A"
    act_calculation_note = ""

    is_video_content_flag = False

    if file_name_for_display.lower().endswith('.mp4') or file_info['name'].lower().endswith('.mp4'): # Check both original and display name for type
        is_video_content_flag = True
        video_full_duration_seconds = get_video_duration(service, file_id)
        ect = video_full_duration_seconds / 60

        base_time_for_act = video_full_duration_seconds
        if video_speech_duration_seconds is not None and video_speech_duration_seconds > 0:
            base_time_for_act = video_speech_duration_seconds
            print(f"  Using speech duration ({video_speech_duration_seconds:.2f}s) from associated audio for ACT/Focus calculation for {file_name_for_display}")
            act_calculation_note = f"Based on {video_speech_duration_seconds:.2f}s of detected speech in associated audio."
        else:
            print(f"  Using full video duration ({video_full_duration_seconds:.2f}s) for ACT/Focus calculation for {file_name_for_display}")
            act_calculation_note = f"Based on full video duration ({video_full_duration_seconds:.2f}s) as no speech audio was detected or provided."

        if transcript_content:
            content_for_analysis = transcript_content
            print(f"  Using provided transcript (length {len(content_for_analysis)} chars) for ACT/Focus calculation for {file_name_for_display}")
        else:
            content_for_analysis = f"This is video content for {file_name_for_display}. No detailed transcript available for semantic analysis."
            print(f"  No transcript provided for {file_name_for_display}. ACT and Focus will be based on generic video content.")

        act, focus_percentage, act_calc_status = calculate_act_and_focus_for_content(
            base_time_for_act,
            content_for_analysis,
            learning_objectives,
            difficulty_level,
            is_video_content=is_video_content_flag
        )
        if act_calc_status: act_calculation_note += " " + act_calc_status


    elif file_name_for_display.lower().endswith('.odt') or file_info['name'].lower().endswith('.odt'):
        ect = calculate_ect(service, file_info, course_readings) # calculate_ect uses file_info['name'] and file_info['id']
        content_for_analysis = extract_from_odt(service, file_id)
        print(f"  Extracted ODT content length: {len(content_for_analysis)} chars.")

        baseline_seconds = calculate_baseline_time(content_for_analysis)
        assignment_cognitive_multiplier, cognitive_load_factors = get_cognitive_load_multiplier_and_factors(
            difficulty_level, "assignment", content_for_analysis
        )
        print(f"  Dynamic assignment cognitive multiplier for {difficulty_level} course: {assignment_cognitive_multiplier:.2f}")
        print(f"  Cognitive Load Factors: {cognitive_load_factors}")

        act, focus_percentage, act_calc_status = calculate_act_and_focus_for_content(
            baseline_seconds * assignment_cognitive_multiplier,
            content_for_analysis,
            learning_objectives,
            difficulty_level,
            is_video_content=False
        )
        if act_calc_status: act_calculation_note += " " + act_calc_status


    elif file_name_for_display.lower().endswith('.pdf') or file_info['name'].lower().endswith('.pdf'): # Only PDF for reading, TXT handled as transcript
        ect = calculate_ect(service, file_info, course_readings) # calculate_ect uses file_info['name'] and file_info['id']
        content_for_analysis = extract_from_pdf(service, file_id)
        print(f"  Extracted PDF content length: {len(content_for_analysis)} chars.")

        baseline_seconds = calculate_baseline_time(content_for_analysis)
        reading_cognitive_multiplier, cognitive_load_factors = get_cognitive_load_multiplier_and_factors(
            difficulty_level, "reading", content_for_analysis
        )
        print(f"  Dynamic reading cognitive multiplier for {difficulty_level} course: {reading_cognitive_multiplier:.2f}")
        print(f"  Cognitive Load Factors: {cognitive_load_factors}")

        act, focus_percentage, act_calc_status = calculate_act_and_focus_for_content(
            baseline_seconds * reading_cognitive_multiplier,
            content_for_analysis,
            learning_objectives,
            difficulty_level,
            is_video_content=False
        )
        if act_calc_status: act_calculation_note += " " + act_calc_status


    elif file_name_for_display.lower().endswith(('.txt', '.wav', '.mp3', '.ogg')) or file_info['name'].lower().endswith(('.txt', '.wav', '.mp3', '.ogg')):
        print(f"  Skipping independent analysis of supporting file: {file_name_for_display}. It's intended for its corresponding media.")
        return None

    else:
        print(f"Skipping unsupported file type: {file_name_for_display}")
        return None

    # Granular TRS and PCI calculation for each file
    trs_score, trs_calc_note = calculate_trs(act, ect)
    trs_score_scaled = trs_score * 4 + 1 if ect > 0 else 1 # Scale 0-1 to 1-5, default to 1 if ECT is 0
    trs_instructor_explanation = get_trs_explanation(trs_score_scaled, ect, act, audience_type="instructor")
    trs_user_explanation = get_trs_explanation(trs_score_scaled, ect, act, audience_type="user")

    pci_score = calculate_pci(trs_score, focus_percentage, course_difficulty_scaling_factor)
    pci_score_scaled = pci_score * 4 + 1 # Scale 0-1 to 1-5
    pci_instructor_explanation = get_pci_explanation(pci_score_scaled, focus_percentage, difficulty_level, ect, act, trs_score, course_difficulty_scaling_factor, audience_type="instructor")
    pci_user_explanation = get_pci_explanation(pci_score_scaled, focus_percentage, difficulty_level, ect, act, trs_score, course_difficulty_scaling_factor, audience_type="user")


    print(f"  Calculated for {file_name_for_display}: ECT={ect:.2f} min, ACT={act:.2f} min, Focus%={focus_percentage:.1f}% ")
    print(f"    TRS: {trs_score_scaled:.2f}/5, PCI: {pci_score_scaled:.2f}/5")

    return {
        'file_name': file_name_for_display, # Use the descriptive name here
        'ect': ect, # Include ECT and ACT for module level summing
        'act': act,
        'focus_percentage': focus_percentage,
        'trs_score_scaled': trs_score_scaled,
        'trs_instructor_explanation': trs_instructor_explanation,
        'trs_user_explanation': trs_user_explanation,
        'pci_score_scaled': pci_score_scaled,
        'pci_instructor_explanation': pci_instructor_explanation,
        'pci_user_explanation': pci_user_explanation,
    }

def analyze_module_recursive(service, folder_id, module_name, learning_objectives, difficulty_level, course_readings, collected_file_results, course_difficulty_scaling_factor, service_account_file, depth=0): # Add service_account_file here
    """
    Recursively analyzes content within a Google Drive module folder and its subfolders.
    """
    indent = "  " * depth
    print(f"{indent}Analyzing folder ID: {folder_id}")

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id, name, mimeType)",
            pageSize=1000
        ).execute()
        items = results.get('files', [])

        if not items:
            print(f"{indent}No items found in this folder.")
            return

        single_mp4 = None
        single_txt = None
        single_audio = None
        other_supported_files = []

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                print(f"{indent}  Entering subfolder: {item['name']} (ID: {item['id']})")
                analyze_module_recursive(
                    service,
                    item['id'],
                    module_name,
                    learning_objectives,
                    difficulty_level,
                    course_readings,
                    collected_file_results,
                    course_difficulty_scaling_factor,
                    service_account_file, # Pass it down
                    depth + 1
                )
            else:
                file_name_original = item['name'] # Store original file name
                file_id = item['id']

                # Get parent folder name and prepend it if not at root or if it's 'index' file
                parent_name = get_parent_folder_name(file_id, service_account_file)

                # Construct a more descriptive file name for display/reporting
                display_file_name = file_name_original # Default to original file name

                # IMPORTANT CHANGE HERE: Set display_file_name to parent_name for content files
                if parent_name and parent_name != 'No parent folder found (possibly in My Drive root)':
                    if file_name_original.lower().endswith(('.mp4', '.odt', '.pdf')):
                        display_file_name = parent_name
                    # For companion files like .txt, .wav, .mp3, .ogg, keep their original name
                    # No else needed, as display_file_name defaults to file_name_original


                # IMPORTANT: When passing file_info, make sure to pass the original file_name and file_id for content extraction,
                # but use display_file_name for the 'file_name' in the results dictionary.
                # This ensures the internal logic of extract_from_pdf, etc., still works with the correct ID.

                file_info_for_processing = {'name': file_name_original, 'id': file_id, 'display_name': display_file_name}

                if file_name_original.lower().endswith('.mp4') and single_mp4 is None:
                    single_mp4 = file_info_for_processing
                elif file_name_original.lower().endswith('.txt') and single_txt is None:
                    single_txt = file_info_for_processing
                elif file_name_original.lower().endswith(('.wav', '.mp3', '.ogg')) and single_audio is None:
                    single_audio = file_info_for_processing
                elif file_name_original.lower().endswith(('.pdf', '.odt')):
                    other_supported_files.append(file_info_for_processing)

        if single_mp4:
            transcript_content = ""
            if single_txt:
                print(f"{indent}  Assuming '{single_txt['display_name']}' is the transcript for '{single_mp4['display_name']}'.")
                transcript_content = extract_from_txt(service, single_txt['id'])

            video_speech_duration_seconds = None
            if single_audio:
                print(f"{indent}  Assuming '{single_audio['display_name']}' is the audio companion for '{single_mp4['display_name']}'.")
                video_speech_duration_seconds = estimate_speech_duration_from_gdrive(service, single_audio['id'])

            file_analysis_result = analyze_file(
                service,
                single_mp4, # Pass the enriched file_info
                learning_objectives,
                difficulty_level,
                course_readings,
                course_difficulty_scaling_factor,
                transcript_content=transcript_content,
                video_speech_duration_seconds=video_speech_duration_seconds
            )
            if file_analysis_result:
                # Update the file_name in the result with the display_name
                file_analysis_result['file_name'] = single_mp4['display_name']
                collected_file_results.append(file_analysis_result)

        for other_file_info in other_supported_files:
            print(f"{indent}  Analyzing independent file: {other_file_info['display_name']}")
            file_analysis_result = analyze_file(
                service,
                other_file_info, # Pass the enriched file_info
                learning_objectives,
                difficulty_level,
                course_readings,
                course_difficulty_scaling_factor
            )
            if file_analysis_result:
                # Update the file_name in the result with the display_name
                file_analysis_result['file_name'] = other_file_info['display_name']
                collected_file_results.append(file_analysis_result)

    except Exception as e:
        print(f"{indent}Error analyzing folder {folder_id}: {e}")


def analyze_module(service, module_folder_id, module_name, learning_objectives, difficulty_level, course_readings, metadata, service_account_file): # Add service_account_file here
    """
    Main function to initiate analysis of a module, including its subfolders.
    """
    print(f"\n=== Analyzing {module_name} (including subfolders) ===")

    course_difficulty_scaling_factor = get_difficulty_scaling(difficulty_level)
    print(f"Course Difficulty Level: {difficulty_level}")

    collected_file_results = []
    analyze_module_recursive(service, module_folder_id, module_name, learning_objectives, difficulty_level, course_readings, collected_file_results, course_difficulty_scaling_factor, service_account_file) # Pass it here

    summed_ect = sum(f['ect'] for f in collected_file_results)
    summed_act = sum(f['act'] for f in collected_file_results)

    total_focus_percentage = sum(f['focus_percentage'] for f in collected_file_results)
    avg_focus_percentage = total_focus_percentage / len(collected_file_results) if collected_file_results else 0.0

    return {
        'module_name': module_name,
        'file_results': collected_file_results,
        'summed_ect': summed_ect,
        'summed_act': summed_act,
        'avg_focus_percentage': avg_focus_percentage,
        'course_difficulty_scaling_factor': course_difficulty_scaling_factor # Add this to module results
    }

def calculate_main_metrics(total_act_minutes, total_ect_minutes, avg_focus_percentage, difficulty_level, metadata, course_difficulty_scaling_factor):
    """Calculates and returns all main metrics for the module."""
    print("\n=== Calculating Main Metrics (Module Level) ===")

    # TRS
    trs_score, _ = calculate_trs(total_act_minutes, total_ect_minutes)
    trs_score_scaled = trs_score * 4 + 1 if total_ect_minutes > 0 else 1.0
    trs_instructor_explanation = get_trs_explanation(trs_score_scaled, total_ect_minutes, total_act_minutes, audience_type="instructor")
    trs_user_explanation = get_trs_explanation(trs_score_scaled, total_ect_minutes, total_act_minutes, audience_type="user")

    # PCI
    pci_score = calculate_pci(trs_score, avg_focus_percentage, course_difficulty_scaling_factor)
    pci_score_scaled = pci_score * 4 + 1
    pci_instructor_explanation = get_pci_explanation(pci_score_scaled, avg_focus_percentage, difficulty_level, total_ect_minutes, total_act_minutes, trs_score, course_difficulty_scaling_factor, audience_type="instructor")
    pci_user_explanation = get_pci_explanation(pci_score_scaled, avg_focus_percentage, difficulty_level, total_ect_minutes, total_act_minutes, trs_score, course_difficulty_scaling_factor, audience_type="user")

    # ETOR (remains module-level)
    etor_value, complexity_score, bloom_classifications_detailed, etor_default_explanation = calculate_etor(total_act_minutes, metadata)

    etor_scaled = 0.0
    etor_instructor_explanation = etor_default_explanation
    etor_user_explanation = etor_default_explanation
    if etor_value > 0:
        min_etor_expected = 0.5
        max_etor_expected = 5.0
        clamped_etor_value = max(min_etor_expected, min(max_etor_expected, etor_value))
        etor_scaled = ((clamped_etor_value - min_etor_expected) / (max_etor_expected - min_etor_expected)) * 4 + 1
        etor_scaled = max(1.0, min(5.0, etor_scaled))
        etor_instructor_explanation = get_etor_explanation(etor_scaled, total_act_minutes, complexity_score, bloom_classifications_detailed, audience_type="instructor")
        etor_user_explanation = get_etor_explanation(etor_scaled, total_act_minutes, complexity_score, bloom_classifications_detailed, audience_type="user")


    return {
        'trs_score_scaled': trs_score_scaled,
        'trs_instructor_explanation': trs_instructor_explanation,
        'trs_user_explanation': trs_user_explanation,
        'pci_score_scaled': pci_score_scaled,
        'pci_instructor_explanation': pci_instructor_explanation,
        'pci_user_explanation': pci_user_explanation,
        'etor_value': etor_value,
        'etor_score_scaled': etor_scaled,
        'etor_instructor_explanation': etor_instructor_explanation,
        'etor_user_explanation': etor_user_explanation,
        'bloom_classifications': bloom_classifications_detailed # Include for JSON output
    }


def display_results(module_results, main_metrics):
    """Displays the calculated results in a formatted way and returns JSON data."""
    # ... (existing print statements for console output, as provided in previous response) ...

    # Prepare data for JSON output (as previously detailed)
    json_output_data = {
        "module_analysis_report": {
            "module_name": module_results['module_name'],
            "module_totals": {
                "total_ect_minutes": round(module_results['summed_ect'], 2), # Corrected to use module_results
                "total_act_minutes": round(module_results['summed_act'], 2), # Corrected to use module_results
                "avg_focus_percentage": round(module_results['avg_focus_percentage'], 1)
            },
            "main_module_metrics": {
                "trs_score": round(main_metrics['trs_score_scaled'], 2),
                "trs_instructor_explanation": main_metrics.get('trs_instructor_explanation', 'N/A'),
                "trs_user_explanation": main_metrics.get('trs_user_explanation', 'N/A'),
                "pci_score": round(main_metrics['pci_score_scaled'], 2),
                "pci_instructor_explanation": main_metrics.get('pci_instructor_explanation', 'N/A'),
                "pci_user_explanation": main_metrics.get('pci_user_explanation', 'N/A'),
                "etor_value": round(main_metrics['etor_value'], 2),
                "Effect to Objective Ratio": round(main_metrics['etor_score_scaled'], 2),
                "instructor_explanation": main_metrics.get('etor_instructor_explanation', 'N/A'),
                "user_explanation": main_metrics.get('etor_user_explanation', 'N/A')
            },
            "learning_objective_bloom_classifications": [],
            "file_by_file_details": []
        }
    }

    if main_metrics.get('bloom_classifications'):
        for bloom_res in main_metrics['bloom_classifications']:
            json_output_data['module_analysis_report']['learning_objective_bloom_classifications'].append({
                "objective": bloom_res['objective'],
                "bloom_level": bloom_res['bloom_level'],
                "bloom_label": bloom_res['bloom_label']
            })

    for file_res in module_results['file_results']:
        file_detail = {
            "file_name": file_res['file_name'],
            "ect_minutes": round(file_res.get('ect', 0.0), 2), # Added ECT to file details
            "act_minutes": round(file_res.get('act', 0.0), 2), # Added ACT to file details
            "focus_percentage": round(file_res.get('focus_percentage', 0.0), 1), # Added Focus % to file details
            "Time Realism Score": round(file_res.get('trs_score_scaled', 0.0), 2),
            "trs_instructor_explanation": file_res.get('trs_instructor_explanation', 'N/A'),
            "trs_user_explanation": file_res.get('trs_user_explanation', 'N/A'),
            "Pace Comfort Index Score": round(file_res.get('pci_score_scaled', 0.0), 2),
            "pci_instructor_explanation": file_res.get('pci_instructor_explanation', 'N/A'),
            "pci_user_explanation": file_res.get('pci_user_explanation', 'N/A')
        }
        json_output_data['module_analysis_report']['file_by_file_details'].append(file_detail)

    # Instead of printing here, return the dictionary
    return json_output_data

# --- Main Execution ---
if __name__ == "__main__":
    SERVICE_ACCOUNT_FILE = r''

    GEMINI_API_KEY = ""
    genai.configure(api_key=GEMINI_API_KEY)
 
    # Replace with your Google Drive folder ID for Module 1
    MODULE_1_FOLDER_ID = ''

    try:
        service = get_gdrive_service(SERVICE_ACCOUNT_FILE)
        print("Google Drive service connected successfully.")
    except Exception as e:
        print(f"Failed to connect to Google Drive service: {e}")
        print("Please ensure 'service_account_key.json' is in the correct directory and has the necessary permissions.")
        exit()

    parent_folder_id = service.files().get(fileId=MODULE_1_FOLDER_ID, fields='parents').execute()['parents'][0]
    metadata = fetch_metadata_json(service, parent_folder_id)

    if not metadata:
        print("Failed to fetch metadata.json. Cannot proceed with analysis.")
        exit()

    course_readings = fetch_course_readings_json(service, parent_folder_id)

    if not course_readings:
        print("Failed to fetch course_readings.json. Proceeding with default ECT for readings.")
        course_readings = []

    difficulty_level = metadata.get('Level', 'Intermediate')
    print(f"Course Difficulty Level: {difficulty_level}")

    learning_objectives = metadata.get('Module 1', {}).get('Learning Objectives', [])
    if not learning_objectives:
        print("Warning: No learning objectives found for 'Module 1' in metadata.")
        print("ACT and ETOR calculations for this module might be less accurate.")

    module_results = analyze_module(
        service,
        MODULE_1_FOLDER_ID,
        "Module 1",
        learning_objectives,
        difficulty_level,
        course_readings,
        metadata,
        SERVICE_ACCOUNT_FILE # Pass the service account file path
    )

    main_metrics = calculate_main_metrics(
        module_results['summed_act'],
        module_results['summed_ect'],
        module_results['avg_focus_percentage'],
        difficulty_level,
        metadata,
        module_results['course_difficulty_scaling_factor']
    )

    # --- Call display_results to print the human-readable report ---

    # Here's how you can save the results to a JSON file:
    all_results_data = display_results(module_results, main_metrics) # This now returns the dict

    output_filename = "taa_analysis_report.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results_data, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis results successfully saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving analysis results to JSON file: {e}")
