from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import os
import json
from scipy.signal import butter, sosfiltfilt
import re
import spacy
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from faster_whisper import WhisperModel
import torch, torchaudio
from torch.utils.data import DataLoader, Dataset
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import google.generativeai as genai
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
import http.client
import tempfile
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import io
import spacy.tokens

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Global model variables for caching
_whisper_model = None
_nlp_model = None
_rouge_scorer = None

def convert_to_serializable(obj):
    # Handle dict
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # Handle list
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    # Handle tuple
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    # Handle set
    elif isinstance(obj, set):
        return [convert_to_serializable(item) for item in obj]
    # Handle native float type with rounding
    elif isinstance(obj, float):
        return round(obj, 2)
    # Handle numpy array, convert and round the output
    elif isinstance(obj, np.ndarray):
        arr = obj.tolist()
        # Recursively round floats inside arrays/lists
        return convert_to_serializable(arr)
    # Handle numpy scalars (e.g., np.float32, np.float64)
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float):
            return round(val, 2)
        return val
    # Handle PyTorch Tensor, convert and round
    elif isinstance(obj, torch.Tensor):
        arr = obj.tolist()
        return convert_to_serializable(arr)
    # Handle PyTorch device
    elif isinstance(obj, torch.device):
        return str(obj)
    # MoviePy VideoFileClip
    elif isinstance(obj, VideoFileClip):
        return f"<VideoFileClip {obj.filename}, duration={round(obj.duration, 2)}s, size={obj.size}>"
    # Google API exceptions
    elif hasattr(obj, '__module__') and 'google' in obj.__module__:
        return repr(obj)
    # HTTP exceptions/objects
    elif hasattr(obj, '__module__') and "http." in obj.__module__:
        return repr(obj)
    # spacy Doc
    try:
        if isinstance(obj, spacy.tokens.doc.Doc):
            return obj.text
    except Exception:
        pass
    # torch Dataset/DataLoader objects
    try:
        if isinstance(obj, torch.utils.data.Dataset):
            return f"<Dataset with {len(obj)} items>"
        if isinstance(obj, torch.utils.data.DataLoader):
            return f"<DataLoader (batch_size={obj.batch_size})>"
    except Exception:
        pass
    # torchaudio audio info (basic)
    try:
        if hasattr(torchaudio, "info") and hasattr(obj, "frame_rate"):
            return f"<AudioObject with frame_rate={obj.frame_rate}>"
    except Exception:
        pass
    if isinstance(obj, (io.IOBase, tempfile._TemporaryFileWrapper)):
        return f"<file-like {type(obj).__name__}>"
    # Exception
    if isinstance(obj, Exception):
        return f"{type(obj).__name__}: {str(obj)}"
    # Any object with __dict__
    if hasattr(obj, "__dict__"):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    # Try JSON, else string
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

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

def find_files_recursively(service, folder_id, extensions, path="", parent_info=None):
    """Find all files with specific extensions in a folder and its subfolders recursively."""
    all_files = []

    # Initialize parent_info if it's None (first call)
    if parent_info is None:
        parent_info = {}

    # Store current folder info in parent_info dictionary
    current_folder_info = {
        'id': folder_id,
        'name': os.path.basename(path) if path else ""
    }

    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()

    items = results.get('files', [])

    for item in items:
        current_path = f"{path}/{item['name']}" if path else item['name']

        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # If it's a folder, recursively search it
            # Pass the updated parent_info to the recursive call
            subfolder_files = find_files_recursively(
                service,
                item['id'],
                extensions,
                current_path,
                {'id': item['id'], 'name': item['name']}  # This folder becomes the parent
            )
            all_files.extend(subfolder_files)
        else:
            # If it's a file with one of the specified extensions, add it to the list
            file_name = item['name']
            if any(file_name.lower().endswith(ext) for ext in extensions):
                # Get directory path and last subfolder name
                directory = os.path.dirname(current_path)

                file_info = {
                    'id': item['id'],
                    'name': item['name'],
                    'path': current_path,
                    'directory': directory,
                    'basename': os.path.splitext(file_name)[0],
                    'extension': os.path.splitext(file_name)[1].lower(),
                    'last_subfolder_id': folder_id
                }
                all_files.append(file_info)

    return all_files

def organize_files_by_module(metadata, all_files):
    # Create module structure
    module_files = {}
    # If metadata exists, use it to organize by module
    if metadata:
        for key, value in metadata.items():
            if key.startswith("Module") and isinstance(value, dict):
                module_name = value.get("Name", key)
                module_files[module_name] = []
                week_files = [each for each in all_files if module_name in each['directory']]
                unique_directories = set(each["directory"] for each in week_files if "directory" in each)
                for each_dir in unique_directories:
                  file_data = {}
                  file_data["directory"] = each_dir
                  for each_file in week_files:
                    if each_file["directory"] == each_dir:
                      file_data["subfolder_id"] = each_file["last_subfolder_id"]
                      if each_file["extension"] == ".vtt":
                        file_data["vtt_file"] = each_file["id"]
                      if each_file["extension"] == ".mp4":
                        file_data["mp4_file"] = each_file["id"]
                      if each_file["extension"] == ".txt":
                        file_data["txt_file"] = each_file["id"]
                      if each_file["extension"] == ".wav":
                        file_data["wav_file"] = each_file["id"]
                  module_files[module_name].append(file_data)
    return module_files

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

def extract_audio_from_video(service_account_file, mp4_file_id, target_folder_id):
    """Convert MP4 to WAV using in-memory processing and store in target folder."""
    service = get_gdrive_service(service_account_file)

    # Get file metadata to determine the name
    file_metadata = service.files().get(fileId=mp4_file_id, fields='name').execute()
    original_filename = file_metadata.get('name', 'video')
    wav_filename = os.path.splitext(original_filename)[0] + '.wav'

    # Create a temporary directory to work with VideoFileClip
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_mp4_path = os.path.join(temp_dir, 'temp_video.mp4')
        temp_wav_path = os.path.join(temp_dir, 'temp_audio.wav')

        # Stream the MP4 file into memory and then to the temporary file
        request = service.files().get_media(fileId=mp4_file_id)
        mp4_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(mp4_buffer, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}%")

        # Write the buffer to the temporary file
        with open(temp_mp4_path, 'wb') as f:
            f.write(mp4_buffer.getvalue())

        # Convert MP4 to WAV using VideoFileClip
        try:
            video_clip = VideoFileClip(temp_mp4_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_wav_path, verbose=False, logger=None)
            audio_clip.close()
            video_clip.close()

            # Read the WAV file into memory
            with open(temp_wav_path, 'rb') as f:
                wav_content = f.read()

            # Upload the WAV content to Google Drive
            wav_buffer = io.BytesIO(wav_content)
            media = MediaIoBaseUpload(wav_buffer, mimetype='audio/wav', resumable=True)

            file_metadata = {
                'name': wav_filename,
                'parents': [target_folder_id]
            }

            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            return uploaded_file.get('id')

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            return None

# Custom bandpass filter using scipy
def bandpass_filter(audio, lowcut, highcut, sr, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=sr, output='sos')
    filtered = np.stack([sosfiltfilt(sos, channel) for channel in audio.cpu().numpy()])
    return torch.from_numpy(filtered).to(audio.device)

# Metrics
def calculate_snr(audio):
    signal_power = torch.mean(audio ** 2)
    noise_power = torch.mean((audio - torch.mean(audio)) ** 2)
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return torch.clamp(snr_db / 60, 0, 1)

def clipping_rate(audio, threshold=0.99):
    clipped = torch.sum(torch.abs(audio) >= threshold)
    return 1 - torch.clamp(clipped / audio.numel(), 0, 1)

def dynamic_range(audio):
    return (torch.max(audio) - torch.min(audio)) / 2

def approximate_sii(audio, sr, bands=[(300,600), (600,1200), (1200,2400), (2400,4800)]):
    total_snr = 0
    for low, high in bands:
        filtered = bandpass_filter(audio, low, high, sr)
        signal_power = torch.mean(filtered ** 2)
        noise_power = torch.mean((filtered - torch.mean(filtered)) ** 2)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        total_snr += torch.clamp(snr, 0, 60)
    return (total_snr / len(bands)) / 60

def detect_echo_reverberation(audio, sr, n_fft=512):
    spec = torch.stft(audio, n_fft=n_fft, return_complex=True).abs()
    env = torch.mean(spec, dim=1)  # average freq bins -> (channels, time_frames)
    env = torch.mean(env, dim=0)   # average channels -> (time_frames,)
    log_env = torch.log(env + 1e-10)
    x = torch.arange(len(log_env), device=audio.device, dtype=torch.float32)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    y = log_env.unsqueeze(1)
    ATA = A.T @ A
    ATy = A.T @ y
    coeffs = torch.linalg.solve(ATA, ATy)
    slope = coeffs[0, 0]
    return 1 - torch.clamp(torch.abs(slope), 0, 1)

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
        # Auth setup
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

def download_file_to_temp(file_id, service_account_file):
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    # Create a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    downloader = MediaIoBaseDownload(temp, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    temp.close()
    return temp.name  # Return the temp file path

def get_audio(file_id, service_account_file):
    temp_path = download_file_to_temp(file_id, service_account_file)
    audio, sr = torchaudio.load(temp_path)
    os.remove(temp_path)  # Delete the temp file after loading
    return audio.squeeze(0), sr

# Dataset
class AudioAnalysisDatasetv1(Dataset):
    def __init__(self, file_list, service_account_file):
        self.files = file_list
        self.service_account_file = service_account_file

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = get_audio(self.files[idx], self.service_account_file)
        file_name = get_parent_folder_name(self.files[idx], self.service_account_file)
        return file_name, audio.squeeze(0), sr

def scale_to_rating(score):
    """
    Converts a score in the range 0-1 to a rating between 1 and 5.
    """
    rating = (score * 4) + 1
    if rating < 1:
        rating = 1
    elif rating > 5:
        rating = 5
    return round(float(rating), 2)

def execute_prompt(prompt):
    prompt += """
    • Use simple, non-technical language that anyone can understand.
    • Do not mention or reference any of the provided metric names or values—these are confidential.
    • Focus on what the data implies or suggests, not on the metric itself.
    """
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = "None"
    for _ in range(3):
        try:
            output = model.generate_content(prompt)
            response = output.text.strip()
            break
        except InternalServerError as e:
            #print("Internal Server Error, retrying...")
            time.sleep(3)
        except ResourceExhausted as r:
            time.sleep(3)
        except Exception as e:
            time.sleep(3)
    print(f"Response: {response}")
    return response

def prompt_signal_quality(intermediate_snr, intermediate_clipping, intermediate_dynrange, final_rating):
    learner_prompt = f"""
    You are simulating a learner's perception of audio quality while evaluating educational content. The **Signal Quality** metric still reflects audio clarity, distortion, and dynamic range, but the output should now resemble how a user might *feel* or *experience* the audio quality.

    **Evaluation Methodology:**
    - **Signal-to-Noise Ratio (SNR)** (0–1): Indicates clarity—a value close to 1 means very little background noise, while values near 0 indicate significant hiss/static.
    - **Clipping Rate** (0–1): Measures distortion—closer to 1 means minimal harsh clipping, while lower values flag more distortion.
    - **Dynamic Range** (0–1): Reflects expressiveness and naturalness—values near 1 mean crisp, lively audio, while low values suggest the audio is flat.
    - **Final Rating**: Aggregated score scaled from 1 (poor) to 5 (excellent).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived clarity, distortion, and naturalness, based on input values.

    **Prompt Instruction:**
    Given SNR: {intermediate_snr}, Clipping Rate: {intermediate_clipping}, Dynamic Range: {intermediate_dynrange}, and Final Rating: {final_rating}, write a short learner-style reflection on the audio quality. Avoid technical jargon. Simulate how a learner *might describe the experience* of listening to the content.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational audio content using the **Signal Quality** metric, which quantifies SNR, Clipping Rate, and Dynamic Range — each ranging from 0 (worst) to 1 (best).

    **Methodology & Overview:**
    - SNR assesses clarity; Clipping Rate evaluates distortion; Dynamic Range reflects how lifelike the audio is.
    - The final score is computed by averaging these metrics and scaling the result to a range from 1 (unacceptable) to 5 (excellent).

    **Score Ranges:**
    - SNR, Clipping Rate, Dynamic Range: Range from 0 (poor) to 1 (ideal).
    - Final Score: Float between 1 (unacceptable) and 5 (excellent), with 5 indicating exceptionally clear and robust audio.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    SNR: {intermediate_snr}, Clipping Rate: {intermediate_clipping}, Dynamic Range: {intermediate_dynrange}, Final Rating: {final_rating}.  
    Please provide concise, 2–4 line feedback identifying which metric deviates most from the ideal value of 1.  
    The response should be in passive voice, offering suggestions on how that specific weakness can be improved.  
    Avoid unnecessary elaboration — keep it formal, focused, and clear.
    """
    return learner_prompt, instructor_prompt

def prompt_signal_quality_batch(intermediate_snr_list, intermediate_clipping_list, intermediate_dynrange_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner's perception of audio quality while evaluating a batch of educational content. The **Signal Quality** metric reflects audio clarity, distortion, and dynamic range, and the output should now resemble how a user might *feel* or *experience* the audio quality across multiple audios.

    **Evaluation Methodology:**
    - **Signal-to-Noise Ratio (SNR)** (0–1): Indicates clarity—a value close to 1 means very little background noise, while values near 0 indicate significant hiss/static.
    - **Clipping Rate** (0–1): Measures distortion—closer to 1 means minimal harsh clipping, while lower values flag more distortion.
    - **Dynamic Range** (0–1): Reflects expressiveness and naturalness—values near 1 mean crisp, lively audio, while low values suggest the audio is flat.
    - **Final Rating**: Aggregated score scaled from 1 (poor) to 5 (excellent).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived clarity, distortion, and naturalness, based on input values.

    Given the following batch-level inputs:
    - SNR values: {intermediate_snr_list}
    - Clipping Rate values: {intermediate_clipping_list}
    - Dynamic Range values: {intermediate_dynrange_list}
    - Final Ratings: {final_rating_list}

    **Prompt Instruction:**
    Write a short learner-style reflection on the audio quality. Avoid technical jargon. Simulate how a learner *might describe the experience* of listening to the content.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating a batch of educational audio samples using the **Signal Quality** metric. Each audio is assessed for SNR (clarity), Clipping Rate (distortion), and Dynamic Range (naturalness), all ranging from 0 (poor) to 1 (ideal).

    **Methodology & Overview:**
    - SNR assesses clarity; Clipping Rate evaluates distortion; Dynamic Range reflects how lifelike the audio is.
    - The final score is computed by averaging these metrics and scaling the result to a range from 1 (unacceptable) to 5 (excellent).

    **Score Ranges:**
    - SNR, Clipping Rate, Dynamic Range: Range from 0 (poor) to 1 (ideal).
    - Final Score: Float between 1 (unacceptable) and 5 (excellent), with 5 indicating exceptionally clear and robust audio.

    Given the following batch-level inputs:
    - SNR values: {intermediate_snr_list}
    - Clipping Rate values: {intermediate_clipping_list}
    - Dynamic Range values: {intermediate_dynrange_list}
    - Final Ratings: {final_rating_list}

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**  
    Please provide concise, 2–4 line feedback identifying which metric deviates most from the ideal value of 1.  
    The response should be in passive voice, offering suggestions on how that specific weakness can be improved.  
    Avoid unnecessary elaboration — keep it formal, focused, and clear.
    """
    return learner_prompt, instructor_prompt

def prompt_signal_quality_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of audio quality while evaluating a **course** composed of multiple modules. The **Signal Quality** metric reflects how clearly and naturally the audio was perceived across the course.

    **Evaluation Methodology:**
    - **Signal-to-Noise Ratio (SNR)** (0–1): Indicates clarity—a value close to 1 suggests minimal background hiss or static.
    - **Clipping Rate** (0–1): Measures distortion—higher values indicate cleaner audio with fewer harsh peaks or breaks.
    - **Dynamic Range** (0–1): Reflects expressiveness and richness—values near 1 suggest lively, natural sound.
    - **Final Rating**: Scaled per module between 1 (poor) and 5 (excellent).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice where natural.
    - Focus: Overall clarity, distortion-free experience, and natural feel across the course.

    **Prompt Instruction:**
    Reflect on the **course-level** audio experience based on the module-level input below:

    {module_level_data}

    Write a learner-style summary expressing how the audio *felt* throughout the course. Avoid technical language and focus on experiential clarity and comfort.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Signal Quality** of a course based on audio data from its modules. This metric integrates:
    
    **Evaluation Methodology:**
    - **Signal-to-Noise Ratio (SNR)** (0–1): Indicates clarity—a value close to 1 suggests minimal background hiss or static.
    - **Clipping Rate** (0–1): Measures distortion—higher values indicate cleaner audio with fewer harsh peaks or breaks.
    - **Dynamic Range** (0–1): Reflects expressiveness and richness—values near 1 suggest lively, natural sound.
    - **Final Rating**: Scaled per module between 1 (poor) and 5 (excellent).

    **Module-Level Data:**
    {module_level_data}

    **Instruction:**
    - Use formal tone and passive voice.
    - Provide a short (2–4 line) summary at the course level.
    - Identify which one among SNR, Clipping Rate, or Dynamic Range most consistently fell short of the ideal (1).
    - Suggest one actionable improvement, without referring to individual modules.

    Avoid elaboration; focus strictly on what's most in need of attention.
    """
    
    return learner_prompt, instructor_prompt

def prompt_speech_clarity(intermediate_silence, intermediate_sii, final_rating):
    learner_prompt = f"""
    You are simulating a learner's perception of speech clarity while engaging with educational spoken content. The **Speech Clarity** metric reflects how intelligible and consistently delivered the speech is, but the output should now resemble how a user might *feel* or *experience* the clarity of speech.

    **Evaluation Methodology:**
    - **Silence Ratio** (0–1): Fraction of time containing speech—1 means nearly always speaking, 0 means silence dominates.
    - **Speech Intelligibility Index (SII)** (0–1): Closer to 1 shows high clarity/understandability; values near 0 indicate listeners will struggle.
    - **Final Rating**: Computed from normalized silence ratio and SII, scaled 1 (hardest to follow) to 5 (clearest).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived speech clarity and consistency, based on input values.

    **Prompt Instruction:**
    Given Silence Ratio: {intermediate_silence}, SII: {intermediate_sii}, and Final Rating: {final_rating}, write a short learner-style reflection on the clarity of speech in the audio. Avoid technical jargon. Simulate how a learner *might describe the experience* of understanding the speaker.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational audio content using the **Speech Clarity** metric, which analyzes Silence Ratio and Speech Intelligibility Index (SII), each ranging from 0 (worst) to 1 (best). Values closer to 1 are ideal for clear and engaging delivery.

    **Methodology & Overview:**
    - Silence Ratio (0–1): Reflects how much of the audio contains speech; values near 1 indicate continuous speaking.
    - Speech Intelligibility Index (SII) (0–1): Indicates how understandable the speech is; 1 reflects maximum intelligibility.

    **Final Score Derivation:**
    - The final score is calculated by averaging the two metrics and scaling the result to a range from 1 (hardest to follow) to 5 (clearest).

    **Score Ranges:**
    - Silence Ratio, SII: Range from 0 (poor) to 1 (ideal).
    - Final Score: Float between 1 (unclear) and 5 (excellent), with 5 representing highly intelligible and consistently spoken content.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Silence Ratio: {intermediate_silence}, SII: {intermediate_sii}, Final Rating: {final_rating}.  
    Please provide a 2–4 line feedback summary that identifies the metric deviating most from 1.  
    The response should be in passive voice and offer focused recommendations for enhancing clarity.  
    Avoid redundancy; keep the tone formal and user-directed.
    """
    return learner_prompt, instructor_prompt

def prompt_speech_clarity_batch(intermediate_silence_list, intermediate_sii_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner's perception of speech clarity while engaging with a **batch of educational audio samples**. The **Speech Clarity** metric reflects how intelligible and consistently delivered the speech is, but the output should now resemble how a user might *feel* or *experience* the clarity of speech across the entire set.

    **Evaluation Methodology:**
    - **Silence Ratio** (0–1): Fraction of time containing speech—1 means nearly always speaking, 0 means silence dominates.
    - **Speech Intelligibility Index (SII)** (0–1): Closer to 1 shows high clarity/understandability; values near 0 indicate listeners will struggle.
    - **Final Rating**: Computed from normalized silence ratio and SII, scaled 1 (hardest to follow) to 5 (clearest).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived speech clarity and consistency, based on input values.

    Given the following batch-level inputs:
    - Silence Ratio values: {intermediate_silence_list}
    - Speech Intelligibility Index (SII) values: {intermediate_sii_list}
    - Final Ratings: {final_rating_list}

    **Prompt Instruction:**
    Write a short learner-style reflection on the clarity of speech in the audio. Avoid technical jargon. Simulate how a learner *might describe the experience* of understanding the speaker.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch of educational audio samples** using the **Speech Clarity** metric. Each sample is analyzed for Silence Ratio and Speech Intelligibility Index (SII), both ranging from 0 (poor) to 1 (ideal). Higher values are preferred for a clear and engaging delivery.

    **Methodology & Overview:**
    - Silence Ratio (0–1): Reflects how much of the audio contains speech; values near 1 indicate continuous speaking.
    - Speech Intelligibility Index (SII) (0–1): Indicates how understandable the speech is; 1 reflects maximum intelligibility.

    **Final Score Derivation:**
    - The final score is calculated by averaging the two metrics and scaling the result to a range from 1 (hardest to follow) to 5 (clearest).

    **Score Ranges:**
    - Silence Ratio, SII: Range from 0 (poor) to 1 (ideal).
    - Final Score: Float between 1 (unclear) and 5 (excellent), with 5 representing highly intelligible and consistently spoken content.

    Given the following batch-level inputs:
    - Silence Ratio values: {intermediate_silence_list}
    - Speech Intelligibility Index (SII) values: {intermediate_sii_list}
    - Final Ratings: {final_rating_list}

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):** 
    Please provide a 2–4 line feedback summary that identifies the metric deviating most from 1.  
    The response should be in passive voice and offer focused recommendations for enhancing clarity.  
    Avoid redundancy; keep the tone formal and user-directed.
    """
    return learner_prompt, instructor_prompt

def prompt_speech_clarity_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of **speech clarity** while engaging with a **course** composed of multiple audio modules. The **Speech Clarity** metric reflects how easily the learner can understand and follow the spoken content throughout the course.

    **Evaluation Methodology:**
    - **Silence Ratio** (0–1): Reflects how much of the audio contains speech. Higher values mean fewer silent gaps.
    - **Speech Intelligibility Index (SII)** (0–1): Indicates how understandable the speech is. Higher values reflect better clarity.
    - **Final Rating**: Computed from normalized silence ratio and SII, scaled between 1 (hardest to follow) and 5 (clearest).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view (not AI evaluator).
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection, preferably in passive voice.
    - Focus: Perceived clarity and consistency of speech across the course, avoiding technical terms.

    **Prompt Instruction:**
    Given the following **module-level data**, write a short reflection that summarizes the **overall clarity of speech** in the course. Simulate how a learner might describe the experience of understanding the spoken content throughout the modules.

    {module_level_data}
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **course** consisting of multiple audio modules using the **Speech Clarity** metric. This metric is based on:
    - **Silence Ratio** (0–1): Measures how much of the audio contains speech. Higher is better.
    - **Speech Intelligibility Index (SII)** (0–1): Measures how understandable the speech is. Higher is better.
    - **Final Rating**: Scaled from 1 (poor clarity) to 5 (excellent clarity), based on the above metrics.

    **Scoring Insight:**
    - Ideal values: Silence Ratio ≈ 1, SII ≈ 1
    - Ratings are derived from the average of these two and scaled accordingly.

    **Module-Level Input Data:**
    {module_level_data}

    **Response Guidelines:**
    - Write a 2–4 line summary using passive voice and a formal tone.
    - Identify the metric (Silence Ratio or SII) that deviated most from its target across modules.
    - Focus on a single actionable suggestion to enhance speech clarity.
    - Avoid module-specific analysis; focus on course-level insight.

    Write a concise, evaluator-style summary offering feedback on the course's overall speech clarity.
    """
    
    return learner_prompt, instructor_prompt

def prompt_audio_aesthetic(intermediate_bgnoise, intermediate_echo, final_rating):
    learner_prompt = f"""
    You are simulating a learner's perception of audio comfort and ambiance while evaluating educational content. The **Audio Aesthetic** metric reflects how pleasing and non-distracting the audio environment feels during listening.

    **Evaluation Methodology:**
    - **Background Noise Presence** (0–1): Lower is better; 0 indicates a noiseless environment, while 1 reflects loud or distracting noise.
    - **Echo/Reverberation Detection** (0–1): Higher values indicate minimal echo and a more professional sound.

    **Final Rating**: Aggregated and scaled between 1 (unpleasant) and 5 (ideal comfort).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice wherever natural.
    - Focus: Perceived comfort, clarity of ambiance, and absence of distractions.

    **Prompt Instruction:**
    Given Background Noise: {intermediate_bgnoise}, Echo: {intermediate_echo}, and Final Rating: {final_rating}, write a short learner-style reflection on the aesthetic experience of the audio. Avoid technical detail. Simulate how a learner might describe their comfort during listening.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational audio content using the **Audio Aesthetic** metric, derived from **Background Noise Presence** (target: 0) and **Echo/Reverberation Detection** (target: 1), both ranging from 0 (poor) to 1 (ideal).

    **Methodology & Overview:**
    - Background Noise Presence (0–1): Lower is better; 0 means noiseless, 1 indicates distracting noise.
    - Echo/Reverberation Detection (0–1): Higher is better; 1 indicates clean, professional acoustics.

    **Final Score Derivation:**
    - The Aesthetic score is calculated by aggregating both metrics and scaling the result to a range from 1 (poor) to 5 (excellent), with 5 reflecting a studio-grade sound environment.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Noise: {intermediate_bgnoise}, Echo: {intermediate_echo}, Final Rating: {final_rating}.  
    The metric furthest from its ideal (Noise → 0, Echo → 1) should be identified.  
    Brief feedback must be given in passive voice, with actionable suggestions to improve the dominant issue.  
    Avoid elaboration and ensure the tone remains formal, direct, and user-friendly.
    """
    return learner_prompt, instructor_prompt

def prompt_audio_aesthetic_batch(intermediate_bgnoise_list, intermediate_echo_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner's perception of audio comfort and ambiance while evaluating a batch of educational content. The **Audio Aesthetic** metric reflects how pleasing and non-distracting the audio environment feels during listening, across all audio samples.

    **Evaluation Methodology:**
    - **Background Noise Presence** (0–1): Lower is better; 0 indicates a noiseless environment, while 1 reflects loud or distracting noise.
    - **Echo/Reverberation Detection** (0–1): Higher values indicate minimal echo and a more professional sound.

    **Final Ratings**: Computed per sample and scaled between 1 (unpleasant) and 5 (ideal comfort).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice wherever natural.
    - Focus: Overall comfort, clarity of ambiance, and absence of distractions across the batch.

    **Prompt Instruction:**
    Given the following batch-level input:
    - Background Noise values: {intermediate_bgnoise_list}
    - Echo/Reverberation values: {intermediate_echo_list}
    - Final Ratings: {final_rating_list}

    Write a short learner-style reflection summarizing the **overall** aesthetic experience of the audio across the batch. Avoid technical detail. Simulate how a learner might describe their comfort during listening to this set of recordings.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch** of educational audio recordings using the **Audio Aesthetic** metric, derived from **Background Noise Presence** (ideal: 0) and **Echo/Reverberation Detection** (ideal: 1). These are measured for each sample on a 0–1 scale, and the final score per sample is scaled between 1 (poor) and 5 (excellent).

    **Evaluation Methodology:**
    - **Background Noise Presence** (0–1): Lower is better; 0 indicates a noiseless environment, while 1 reflects loud or distracting noise.
    - **Echo/Reverberation Detection** (0–1): Higher values indicate minimal echo and a more professional sound.

    **Final Ratings**: Computed per sample and scaled between 1 (unpleasant) and 5 (ideal comfort).

    **Methodology & Overview:**
    - Background Noise values: {intermediate_bgnoise_list}
    - Echo/Reverberation values: {intermediate_echo_list}
    - Final Ratings (1–5): {final_rating_list}

    Analyze the batch-level performance and identify which metric consistently deviated most from its target.  
    Provide a concise (2–4 lines) summary using passive voice and formal tone.  
    Focus on actionable suggestions to improve the most prevalent aesthetic issue. Avoid sample-specific commentary.
    """
    return learner_prompt, instructor_prompt

def prompt_audio_aesthetic_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of audio comfort and ambiance while evaluating a **course** made up of multiple modules. The **Audio Aesthetic** metric reflects how pleasing and non-distracting the audio environment feels during listening.

    **Evaluation Methodology:**
    - **Background Noise Presence** (0–1): Lower is better; 0 indicates a noiseless environment, while 1 reflects loud or distracting noise.
    - **Echo/Reverberation Detection** (0–1): Higher values indicate minimal echo and a more professional sound.
    - **Final Ratings**: Computed per module and scaled between 1 (unpleasant) and 5 (ideal comfort).

    **Expected Response Guidelines:**
    - Perspective: From a learner's simulated point of view (not AI evaluator).
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice wherever natural.
    - Focus: Overall comfort, clarity of ambiance, and absence of distractions throughout the course.

    **Prompt Instruction:**
    Given the following **module-level input**, reflect on the overall **course-level** listening experience, focusing on audio comfort, ambiance, and distraction:

    {module_level_data}

    Write a short learner-style reflection summarizing the **overall** audio aesthetic experience of the course. Avoid technical jargon. Simulate how a learner might describe their comfort during listening across the course.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Audio Aesthetic** quality of a course comprising multiple modules. The metric reflects the listening environment's comfort based on:
    - **Background Noise Presence** (0–1): Lower is better.
    - **Echo/Reverberation Detection** (0–1): Higher is better.
    - **Final Ratings**: Scaled between 1 (poor) and 5 (excellent) per module.

    **Objective:**
    Assess the overall performance across all modules and determine which factor—background noise or echo—most consistently deviated from its target. Provide course-level insights.

    **Module-Level Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Summarize the dominant aesthetic issue affecting course quality.
    - Focus on one actionable area for improvement.
    - Avoid module-specific breakdowns.

    Write a brief (2–4 line) instructor-style summary evaluating the **course-wide** audio aesthetic performance and improvement suggestions.
    """
    return learner_prompt, instructor_prompt

# Batch analyzer
def analyze_batch_1(batch):
    results = []
    with torch.no_grad():
        for file_name, audio, sr in batch:
            audio = audio.to(device)
            audio = audio / torch.max(torch.abs(audio) + 1e-9)

            # Signal Quality
            snr = calculate_snr(audio)
            clip = clipping_rate(audio)
            dyn = dynamic_range(audio)
            signal_score = (snr + clip + dyn) / 3

            # Speech Clarity
            silence = 1 - (torch.mean((torch.abs(audio) < 0.01).float()))
            sii = approximate_sii(audio, sr)
            speech_score = (silence + sii) / 2

            # Audio Aesthetic
            bg_noise = 1 - torch.mean((torch.abs(audio) < 0.02).float())
            echo = detect_echo_reverberation(audio, sr)
            aesthetic_score = (bg_noise + echo) / 2

            signal_score = scale_to_rating(signal_score)
            speech_score = scale_to_rating(speech_score)
            aesthetic_score = scale_to_rating(aesthetic_score)

            signal_score_user_feedback = execute_prompt(prompt_signal_quality(snr, clip, dyn, signal_score)[0])
            signal_score_instructor_feedback = execute_prompt(prompt_signal_quality(snr, clip, dyn, signal_score)[1])
            speech_score_user_feedback = execute_prompt(prompt_speech_clarity(silence, sii, speech_score)[0])
            speech_score_instructor_feedback = execute_prompt(prompt_speech_clarity(silence, sii, speech_score)[1])
            aesthetic_score_user_feedback = execute_prompt(prompt_audio_aesthetic(bg_noise, echo, aesthetic_score)[0])
            aesthetic_score_instructor_feedback = execute_prompt(prompt_audio_aesthetic(bg_noise, echo, aesthetic_score)[1])

            results.append({
                "File": file_name,
                "Signal Quality Score": {"Intermediate Parameters": {"Signal Quality Ratio": snr, "Clipping Rate": clip, "Dynamic Range": dyn}, "Final Score": signal_score, "Learner Perspective Assessment": signal_score_user_feedback, "Instructor Feedback": signal_score_instructor_feedback},
                "Speech Clarity Score": {"Intermediate Parameters": {"Silence Ratio": silence, "Speech Intelligibility Index": sii}, "Final Score": speech_score, "Learner Perspective Assessment": speech_score_user_feedback, "Instructor Feedback": speech_score_instructor_feedback},
                "Audio Aesthetic Score": {"Intermediate Parameters": {"Background Noise": bg_noise, "Echo Reverberation": echo}, "Final Score": aesthetic_score, "Learner Perspective Assessment": aesthetic_score_user_feedback, "Instructor Feedback": aesthetic_score_instructor_feedback}
            })

    return results

def save_dict_to_json(data: dict, file_path: str):
    """
    Saves a dictionary to a JSON file.

    Parameters:
    - data: dict, the dictionary to save
    - file_path: str, path to the output .json file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def perform_audio_analysis(service_account_file, folder_id):
    result = {}
    result["Course Name"] = get_folder_name(folder_id, service_account_file)
    service = get_gdrive_service(service_account_file)
    # Try to fetch metadata.json
    metadata = fetch_metadata_json(service, folder_id)
    extensions = ['.wav', '.mp4']
    all_files = find_files_recursively(service, folder_id, extensions)
    module_files = organize_files_by_module(metadata, all_files)
    audio_summaries = []
    total_results = []
    count = 0
    for key, value in metadata.items():
        if count >= 2:
          break
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            #video_sub_folder_files = [(d["mp4_file"], d["subfolder_id"]) for d in module_files[module_name] if "wav_file" in d and "subfolder_id" in d ]
            #Parallel(n_jobs=os.cpu_count()-1)(delayed(extract_audio_from_video)(service_account_file, file_path, sub_folder) for file_path, sub_folder in video_sub_folder_files)
            audio_files = [d["wav_file"] for d in module_files[module_name] if "wav_file" in d][:2]
            dataset = AudioAnalysisDatasetv1(audio_files, service_account_file)
            loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x, num_workers=os.cpu_count()-1)
            module_results = {}
            module_results["Name"] = module_name
            module_results["Results"] = []
            for batch in loader:
                temp_ = analyze_batch_1(batch)
                module_results["Results"].extend(temp_)
                audio_summaries.extend(temp_)
            module_results["Signal Quality Score"] = {}
            signal_quality_ratings = [each["Signal Quality Score"]["Final Score"] for each in module_results["Results"]]
            signal_quality_ratio_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Signal Quality Ratio"] for each in module_results["Results"]]
            clipping_rate_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Clipping Rate"] for each in module_results["Results"]]
            dynamic_range_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Dynamic Range"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_signal_quality_batch(signal_quality_ratio_list, clipping_rate_list, dynamic_range_list, signal_quality_ratings)
            module_results["Signal Quality Score"]["Score"] = round(sum(signal_quality_ratings)/len(signal_quality_ratings), 2)
            module_results["Signal Quality Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Signal Quality Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results["Speech Clarity Score"] = {}
            speech_clarity_ratings = [each["Speech Clarity Score"]["Final Score"] for each in module_results["Results"]]
            silence_ratio_list = [each["Speech Clarity Score"]["Intermediate Parameters"]["Silence Ratio"] for each in module_results["Results"]]
            speech_intelligibility_index_list = [each["Speech Clarity Score"]["Intermediate Parameters"]["Speech Intelligibility Index"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_speech_clarity_batch(silence_ratio_list, speech_intelligibility_index_list, speech_clarity_ratings)
            module_results["Speech Clarity Score"]["Score"] = round(sum(speech_clarity_ratings)/len(speech_clarity_ratings), 2)
            module_results["Speech Clarity Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Speech Clarity Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results["Audio Aesthetic Score"] = {}
            audio_aesthetic_ratings = [each["Audio Aesthetic Score"]["Final Score"] for each in module_results["Results"]]
            background_noise_list = [each["Audio Aesthetic Score"]["Intermediate Parameters"]["Background Noise"] for each in module_results["Results"]]
            echo_reverberation_list = [each["Audio Aesthetic Score"]["Intermediate Parameters"]["Echo Reverberation"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_audio_aesthetic_batch(background_noise_list, echo_reverberation_list, audio_aesthetic_ratings)
            module_results["Audio Aesthetic Score"]["Score"] = round(sum(audio_aesthetic_ratings)/len(audio_aesthetic_ratings), 2)
            module_results["Audio Aesthetic Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Audio Aesthetic Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            total_results.append(module_results)
            count += 1
    result["Module Results"] = total_results
    result["Course Results"] = {}
    result["Course Results"]["Signal Quality Score"] = {}
    result["Course Results"]["Speech Clarity Score"] = {}
    result["Course Results"]["Audio Aesthetic Score"] = {}
    signal_quality_parameter = ""
    speech_clarity_parameter = ""
    audio_aesthetic_parameter = ""
    final_signal_quality_scores = []
    final_speech_clarity_scores = []
    final_audio_aesthetic_scores = []
    for module_result in result["Module Results"]:
        module_name = module_result["Name"]
        signal_quality_ratings = [each["Signal Quality Score"]["Final Score"] for each in module_result["Results"]]
        signal_quality_ratio_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Signal Quality Ratio"] for each in module_result["Results"]]
        clipping_rate_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Clipping Rate"] for each in module_result["Results"]]
        dynamic_range_list = [each["Signal Quality Score"]["Intermediate Parameters"]["Dynamic Range"] for each in module_result["Results"]]
        signal_quality_parameter += (
            f"Module Name: {module_name}\n"
            f"SNR Values: {signal_quality_ratio_list}\n"
            f"Clipping Rate Values: {clipping_rate_list}\n"
            f"Dynamic Range Values: {dynamic_range_list}\n"
            f"Final Ratings: {signal_quality_ratings}\n"
        )
        final_signal_quality_scores.append(module_result["Signal Quality Score"]["Score"])
        speech_clarity_ratings = [each["Speech Clarity Score"]["Final Score"] for each in module_result["Results"]]
        silence_ratio_list = [each["Speech Clarity Score"]["Intermediate Parameters"]["Silence Ratio"] for each in module_result["Results"]]
        speech_intelligibility_index_list = [each["Speech Clarity Score"]["Intermediate Parameters"]["Speech Intelligibility Index"] for each in module_result["Results"]]
        speech_clarity_parameter += (
            f"Module Name: {module_name}\n"
            f"Silence Ratio values: {silence_ratio_list}\n"
            f"Speech Intelligibility Index (SII) values: {speech_intelligibility_index_list}\n"
            f"Final Ratings: {speech_clarity_ratings}\n"
        )
        final_speech_clarity_scores.append(module_result["Speech Clarity Score"]["Score"])
        audio_aesthetic_ratings = [each["Audio Aesthetic Score"]["Final Score"] for each in module_result["Results"]]
        background_noise_list = [each["Audio Aesthetic Score"]["Intermediate Parameters"]["Background Noise"] for each in module_result["Results"]]
        echo_reverberation_list = [each["Audio Aesthetic Score"]["Intermediate Parameters"]["Echo Reverberation"] for each in module_result["Results"]]
        audio_aesthetic_parameter += (
            f"Module Name: {module_name}\n"
            f"Background Noise Values: {background_noise_list}\n"
            f"Echo/Reverberation values: {echo_reverberation_list}\n"
            f"Final Ratings: {audio_aesthetic_ratings}\n"
        )
        final_audio_aesthetic_scores.append(module_result["Audio Aesthetic Score"]["Score"])    
    learner_prompt, instructor_prompt = prompt_signal_quality_course(signal_quality_parameter)
    result["Course Results"]["Signal Quality Score"]["Score"] = round(sum(final_signal_quality_scores)/len(final_signal_quality_scores),2)
    result["Course Results"]["Signal Quality Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Signal Quality Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_speech_clarity_course(speech_clarity_parameter)
    result["Course Results"]["Speech Clarity Score"]["Score"] = round(sum(final_speech_clarity_scores)/len(final_speech_clarity_scores),2)
    result["Course Results"]["Speech Clarity Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Speech Clarity Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_audio_aesthetic_course(audio_aesthetic_parameter)
    result["Course Results"]["Audio Aesthetic Score"]["Score"] = round(sum(final_audio_aesthetic_scores)/len(final_audio_aesthetic_scores),2)
    result["Course Results"]["Audio Aesthetic Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Audio Aesthetic Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    save_dict_to_json(convert_to_serializable(result), "Audio Quality Results.json")

def prompt_timing_alignment(start_diffs, end_diffs, max_acceptable_diff, timing_alignment_score):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle timing while watching educational content. The **Timing Alignment** metric reflects how well subtitles are synchronized with the spoken audio.

    **Evaluation Methodology:**
    - **Start/End Time Differences**: Ranges are in seconds (≥0); lower is better. Values under the max acceptable difference ({max_acceptable_diff}s) indicate precise alignment.
    - **Final Rating**: Derived by combining and normalizing timing gaps, scaled to a 1–5 score (5 = perfectly synced, 1 = poorly aligned).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived sync of subtitles with audio, based on input values.

    **Prompt Instruction:**
    Given Start Diff: {str(start_diffs)}s, End Diff: {str(end_diffs)}s, Max Acceptable: {str(max_acceptable_diff)}s, and Final Score: {timing_alignment_score}, write a short learner-style reflection on subtitle timing. Avoid technical details. Reflect how naturally subtitles kept pace with the spoken content.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational audio content using the **Timing Alignment** metric, which measures average start and end timing differences (in seconds), normalized and scaled from 1 (poor) to 5 (excellent).

    **Methodology & Overview:**
    - Start and End Time Differences (≥0 s): Lower values indicate better alignment. Values within the acceptable threshold ({max_acceptable_diff}s) are considered precise.
    - Final score is derived by combining and normalizing these differences, then scaling the result to a 1–5 range.

    **Score Ranges:**
    - Start/End Time Differences: ≥0 s, ideally close to 0 s.
    - Final Score: Float from 1 (bad sync) to 5 (perfect alignment).

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Start: {str(start_diffs)}s, End: {str(end_diffs)}s, Max Allowed: {str(max_acceptable_diff)}s, Score: {timing_alignment_score}.  
    Identify where timing deviation exceeds the ideal threshold and advise minimal, precise timestamp corrections.  
    Mention how such discrepancies may affect the user’s comprehension or experience.  
    Keep the tone formal, focused, and user-facing.
    """
    return learner_prompt, instructor_prompt

def prompt_timing_alignment_batch(start_diffs, end_diffs, max_acceptable_diff, timing_alignment_scores):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle timing while watching a **batch of educational content**. The **Timing Alignment** metric reflects how well subtitles across the set were synchronized with the spoken audio.

    **Evaluation Methodology:**
    - **Start/End Time Differences**: Listed in seconds (≥0). Lower values indicate better sync. Values under the max acceptable threshold ({max_acceptable_diff}s) reflect precise alignment.
    - **Final Ratings**: Normalized across the batch and scaled to a 1–5 score (5 = perfectly synced, 1 = poorly aligned).

    **Expected Response Guidelines:**
    - Perspective: Learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence holistic reflection using mostly passive voice.
    - Focus: General experience of subtitle-audio timing across all samples.

    **Prompt Instruction:**
    Given the following:
    - Start Time Differences (s): {start_diffs}
    - End Time Differences (s): {end_diffs}
    - Max Acceptable Diff(s): {max_acceptable_diff}
    - Final Ratings (1–5): {timing_alignment_scores}

    Write a learner-style reflection summarizing the **overall sense of subtitle timing**. Focus on how natural or distracting the pacing felt across the batch. Avoid metric explanations — reflect emotional and cognitive ease or disruption during viewing.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch of educational audios** using the **Timing Alignment** metric. This reflects how well subtitle timings align with speech.
    
    **Evaluation Methodology:**
    - **Start/End Time Differences**: Listed in seconds (≥0). Lower values indicate better sync. Values under the max acceptable threshold ({max_acceptable_diff}s) reflect precise alignment.
    - **Final Ratings**: Normalized across the batch and scaled to a 1–5 score (5 = perfectly synced, 1 = poorly aligned).

    **Metric Overview:**
    - **Start Time Differences** (s): {start_diffs}
    - **End Time Differences** (s): {end_diffs}
    - **Acceptable Threshold** (s): {max_acceptable_diff}
    - **Final Ratings (1–5)**: {timing_alignment_scores}

    **Instruction:**
    Analyze the timing alignment across the batch and identify **systemic misalignments**, if any.  
    Focus on where start or end times frequently exceed the acceptable threshold.  
    In 2–4 lines, write a formal, passive voice summary offering **general corrective recommendations**. Avoid per-sample breakdown — keep it macro and improvement-focused.
    """
    return learner_prompt, instructor_prompt

def prompt_timing_alignment_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle timing while watching a **course** composed of multiple educational modules. The **Timing Alignment** metric reflects how well subtitles were synchronized with the spoken audio throughout the course.

    **Evaluation Methodology:**
    - **Start/End Time Differences**: Measured in seconds (≥0). Lower values indicate better alignment. Timings within the acceptable threshold indicate high synchronization.
    - **Final Ratings**: Calculated per module, scaled from 1 (poor sync) to 5 (excellent alignment).

    **Expected Response Guidelines:**
    - Perspective: Learner’s simulated point of view (not technical).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary using passive voice wherever natural.
    - Focus: Comfort, pacing, and distraction level across the course.

    **Prompt Instruction:**
    Given the following **module-level data**, reflect on the **course-wide** subtitle timing experience:

    {module_level_data}

    Write a short learner-style reflection summarizing how subtitle timing **felt** throughout the course. Avoid technical terms and explain how smooth or distracting the viewing experience was in terms of pacing.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Timing Alignment** across a course composed of several modules. This metric reflects how well subtitle start and end times align with the corresponding speech.

    **Evaluation Methodology:**
    - **Start/End Time Differences**: Measured in seconds (≥0). Lower values represent better sync. Deviations above the acceptable threshold indicate misalignment.
    - **Final Ratings**: Per-module ratings scaled from 1 (poor) to 5 (perfectly aligned).

    **Course Timing Data:**
    {module_level_data}

    **Response Guidelines:**
    - Write in formal tone and passive voice.
    - Summarize timing issues across the course (2–4 lines).
    - Identify whether misalignments were frequent, systemic, or tolerable.
    - Provide **macro-level** improvement suggestions (e.g., consistent shift in start times, late subtitle cutoffs).

    Write a course-level evaluation of subtitle timing quality and recommend general corrective action if needed.
    """
    return learner_prompt, instructor_prompt

def prompt_semantic_match(bleu_score, rouge_score, bert_score, semantic_match_score):
    learner_prompt = f"""
    You are simulating a learner's perception of how accurately subtitles reflect the spoken content in educational material. The **Semantic Match** metric reflects how faithfully the meaning has been preserved.

    **Evaluation Methodology:**
    - **BLEU, ROUGE-1, BERTScore**: Each ranges from 0 to 1, where values closer to 1 indicate better semantic and textual alignment with spoken words.
    - **Final Rating**: Weighted and scaled to a 1–5 score (5 = meaning fully preserved, 1 = significant semantic drift or error).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived accuracy and clarity of subtitle meaning based on provided metric values.

    **Prompt Instruction:**
    Given BLEU: {bleu_score}, ROUGE: {rouge_score}, BERT: {bert_score}, and Final Score: {semantic_match_score}, write a short learner-style reflection on subtitle meaning accuracy. Avoid metric explanations. Focus on how well the intended message seemed to come through.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating subtitle quality using the **Semantic Match** metric, which combines surface-level and deep semantic similarities — each ranging from 0 (poor) to 1 (ideal), with the final score scaled to 1–5.

    **Methodology & Overview:**
    - BLEU, ROUGE-1, and BERTScore: Measure n-gram overlap and semantic preservation. Ideal scores are close to 1.
    - Final score is derived by weighted aggregation of these metrics and scaled to a range from 1 (significant loss) to 5 (meaning perfectly preserved).

    **Score Ranges:**
    - BLEU, ROUGE, BERTScore: 0–1 (ideal = 1)
    - Final Score: Float from 1 (poor match) to 5 (excellent preservation)

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    BLEU: {bleu_score}, ROUGE: {rouge_score}, BERT: {bert_score}, Score: {semantic_match_score}.  
    Call out the metric deviating most from ideal and suggest adjustments to improve meaning retention or textual fidelity.  
    Recommendations should be formal, minimal, and aimed at refining subtitle phrasing or accuracy.
    """
    return learner_prompt, instructor_prompt

def prompt_semantic_match_batch(bleu_scores, rouge_scores, bert_scores, semantic_match_scores):
    learner_prompt = f"""
    You are simulating a learner's perception of how accurately subtitles reflect the spoken content in **a batch of educational audios**. The **Semantic Match** metric reflects how faithfully the meaning has been preserved throughout.

    **Evaluation Methodology:**
    - **BLEU, ROUGE-1, BERTScore**: Each ranges from 0 to 1, where values closer to 1 indicate better semantic and textual alignment with spoken words.
    - **Final Ratings**: Weighted and scaled to a 1–5 score for each audio (5 = meaning fully preserved, 1 = significant semantic drift or error).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence holistic summary, using mostly passive voice.
    - Focus: General perception of meaning preservation across the batch of subtitles, based on the given metric values.

    **Prompt Instruction:**
    Given the following values across the batch:
    - BLEU scores: {bleu_scores}
    - ROUGE scores: {rouge_scores}
    - BERT scores: {bert_scores}
    - Final Semantic Match Ratings: {semantic_match_scores}

    Write a short learner-style reflection describing the **overall perception** of subtitle meaning accuracy across the entire set of audios. Avoid technical explanations. Focus on how well the **intended meaning consistently came through**.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the semantic integrity of subtitles for a **batch** of educational audios using the **Semantic Match** metric.

    **Metric Overview:**
    - **BLEU, ROUGE-1, BERTScore** (0–1): Measure lexical overlap and deep semantic similarity between spoken content and subtitles. Ideal = 1.
    - **Final Ratings**: Weighted and scaled per sample to a 1–5 range (5 = excellent meaning preservation).

    **Batch Values:**
    - BLEU: {bleu_scores}
    - ROUGE: {rouge_scores}
    - BERT: {bert_scores}
    - Final Ratings: {semantic_match_scores}

    **Instruction:**
    Based on these batch-level values, summarize the **overall quality of semantic alignment** between subtitles and audio.  
    Identify the **metric with the greatest deviation** from ideal across the set, and offer brief, improvement-focused guidance.  
    Your tone should be formal, passive, and concise (2–4 lines). Avoid per-audio commentary — focus on macro-level trends.
    """
    return learner_prompt, instructor_prompt

def prompt_semantic_match_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of how accurately subtitles reflect the spoken content across an entire **educational course** comprising multiple modules. The **Semantic Match** metric reflects how faithfully the intended meaning has been preserved in subtitles throughout.

    **Evaluation Methodology:**
    - **BLEU, ROUGE-1, BERTScore**: Each ranges from 0 to 1, where values closer to 1 indicate better semantic and textual alignment with the spoken words.
    - **Final Ratings**: Weighted and scaled per module into a 1–5 range (5 = fully preserved meaning, 1 = major drift or misinterpretation).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence holistic summary, mostly in passive voice.
    - Focus: How well the meaning was consistently preserved throughout the course, as perceived by a learner.

    **Prompt Instruction:**
    Given the following **module-wise semantic match data**, reflect on the **course-wide** experience regarding subtitle accuracy and meaning retention:

    {module_level_data}

    Write a short learner-style reflection summarizing the **overall perception** of semantic alignment across the course. Avoid metric names and technical explanations. Focus purely on meaning retention and clarity as felt during listening.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **semantic accuracy of subtitles** for a course composed of multiple modules using the **Semantic Match** metric.

    **Metric:**
    - **BLEU, ROUGE-1, BERTScore** (0–1): Measure lexical overlap and deep semantic fidelity between audio and subtitle. Ideal = 1.
    - **Final Ratings**: Scaled per module into a 1–5 range (5 = excellent preservation of meaning).

    **Module-Wise Data:**
    {module_level_data}

    **Instruction:**
    Based on the module-level inputs, summarize the **overall semantic quality** of the course.  
    Identify the metric that **most frequently or significantly deviated** from the ideal value, and suggest a concise, course-level improvement direction.  
    Use a formal, passive tone in 2–4 lines. Avoid discussing individual modules — focus on overarching trends and improvement opportunities.
    """
    return learner_prompt, instructor_prompt

def prompt_content_coverage(missing_ratio, extra_ratio, content_coverage_score):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle completeness while engaging with educational content. The **Content Coverage** metric reflects how fully the subtitles capture all that was spoken.

    **Evaluation Methodology:**
    - **Missing Ratio / Extra Ratio**: Each ranges from 0 to 1 (ideal near 0). Lower values suggest fewer omissions or unnecessary additions in subtitles.
    - **Final Rating**: Aggregated from both ratios and scaled to a 1–5 score (5 = fully aligned, 1 = highly incomplete).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Perceived completeness and accuracy of subtitles, based on the input values.

    **Prompt Instruction:**
    Given Missing: {missing_ratio}, Extra: {extra_ratio}, and Final Score: {content_coverage_score}, write a short learner-style reflection on how well the subtitles captured the spoken content. Avoid technical detail. Reflect whether anything felt left out or unnecessarily added.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational content using the **Content Coverage** metric, which considers missing and extra word ratios to assess subtitle completeness. Each ratio ranges from 0 (ideal) to 1 (poor), with the final score scaled from 1 to 5.

    **Methodology & Overview:**
    - Missing and Extra Ratios (0–1): Lower values indicate better alignment between audio and subtitles. Ideal values are near 0.
    - Final score is derived by combining both ratios and scaling the result to a range of 1 (poor coverage) to 5 (complete coverage).

    **Score Ranges:**
    - Missing/Extra Ratios: 0 (ideal) to 1 (worst).
    - Final Score: Float from 1 (incomplete) to 5 (fully accurate).

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Missing: {missing_ratio}, Extra: {extra_ratio}, Final Score: {content_coverage_score}.  
    Highlight which ratio deviates most from ideal, and suggest minimizing subtitle omissions or excesses.  
    Mention the potential impact on learner comprehension and advise refinement for better alignment.
    """
    return learner_prompt, instructor_prompt

def prompt_content_coverage_batch(missing_ratios, extra_ratios, content_coverage_scores):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle completeness while engaging with a **batch of educational content**. The **Content Coverage** metric reflects how fully the subtitles capture all that was spoken across the set.

    **Evaluation Methodology:**
    - **Missing Ratio / Extra Ratio** (each 0–1): Lower values suggest better subtitle alignment. Values near 0 indicate minimal omissions or unnecessary insertions.
    - **Final Ratings**: Scaled from 1 (highly incomplete) to 5 (fully aligned), based on aggregated coverage indicators for each sample.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence reflection.
    - Focus: Overall perception of completeness and subtitle trustworthiness across the batch.

    **Prompt Instruction:**
    Based on the following data:
    - Missing Ratios: {missing_ratios}
    - Extra Ratios: {extra_ratios}
    - Final Content Coverage Ratings: {content_coverage_scores}

    Write a short learner-style reflection summarizing the **overall experience** regarding subtitle completeness. Did the subtitles consistently capture everything that was spoken? Did anything feel left out or awkwardly added? Reflect the general experience without referencing individual scores.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch of educational subtitle tracks** using the **Content Coverage** metric.

    **Metric Summary:**
    - **Missing Ratios** (ideal = 0): {missing_ratios}
    - **Extra Ratios** (ideal = 0): {extra_ratios}
    - **Final Scores** (scaled 1–5): {content_coverage_scores}

    **Evaluation Methodology:**
    - Lower missing/extra ratios imply better subtitle-audio alignment.
    - Final scores represent combined deviation and are scaled to reflect subtitle completeness.

    **Instruction:**
    Write a 2–4 line batch-level diagnostic summary.  
    Identify which ratio (missing or extra) **most consistently deviated from ideal** across the samples.  
    Offer brief, actionable guidance to improve coverage — e.g., reducing omissions or trimming excess text — while keeping the tone formal and passive.
    """
    return learner_prompt, instructor_prompt

def prompt_content_coverage_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of subtitle completeness while engaging with a **course** composed of multiple modules. The **Content Coverage** metric reflects how fully the subtitles capture the spoken content throughout the course.

    **Evaluation Methodology:**
    - **Missing Ratio / Extra Ratio** (each 0–1): Lower values suggest better subtitle alignment. Values close to 0 imply minimal omissions or unnecessary insertions.
    - **Final Ratings**: Scaled from 1 (highly incomplete) to 5 (fully aligned), based on module-level subtitle coverage.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection.
    - Focus: General experience of subtitle trustworthiness and completeness throughout the course.

    **Prompt Instruction:**
    Given the following **module-level subtitle data**, reflect on your **overall course-level experience** with subtitle completeness. Consider whether the subtitles consistently captured what was said, or whether anything felt left out or awkwardly included.

    {module_level_data}

    Write a short learner-style reflection describing the overall comfort and trust in the subtitles used across the course. Avoid technical details or module-specific references.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating subtitle coverage quality across a course made of several modules. The evaluation is based on:

    - **Missing Ratios** (ideal = 0): Indicates portions of spoken content not captured.
    - **Extra Ratios** (ideal = 0): Indicates content in the subtitles not present in the audio.
    - **Final Content Coverage Ratings** (1–5): Reflects completeness and alignment on a module level.

    **Module-Level Data Provided:**
    {module_level_data}

    **Instruction:**
    Write a concise (2–4 lines) instructor-style summary evaluating the course-level subtitle quality.  
    Identify whether **missing** or **extra** ratios most consistently deviated from ideal.  
    Provide one actionable recommendation to improve subtitle alignment and completeness. Avoid naming specific modules.
    """

    return learner_prompt, instructor_prompt

def prompt_entity_accuracy(asr_entities, vtt_entities, entity_accuracy_score):
    learner_prompt = f"""
    You are simulating a learner's perception of content reliability based on subtitle accuracy. The **Entity Accuracy** metric reflects how well subtitles preserve key names, terms, and references from the original audio.

    **Evaluation Methodology:**
    - **ASR Entities / VTT Entities**: Sets of named entities (e.g., persons, places, organizations) detected from audio and subtitles. Greater overlap suggests higher accuracy.
    - **Final Rating**: Based on the degree of entity match, scaled to a 1–5 score (5 = perfect alignment, 1 = frequent mismatches).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Trust and clarity of key terms in subtitles based on match with spoken audio.

    **Prompt Instruction:**
    Given ASR Entities: {asr_entities}, VTT Entities: {vtt_entities}, and Final Score: {entity_accuracy_score}, write a short learner-style reflection on whether key information in the subtitles felt preserved and trustworthy. Avoid technical explanations. Reflect on how clearly named concepts and references were conveyed.
    """
    instructor_prompt = f"""
    You are an AI assistant evaluating educational audio content using the **Entity Accuracy** metric, which measures overlap and precision of named entities between transcript and subtitle. The final score ranges from 1 (inaccurate) to 5 (perfect alignment).

    **Methodology & Overview:**
    - ASR Entities and VTT Entities refer to named entities (e.g., people, places, organizations) extracted from audio and subtitle text.
    - The degree of overlap between these sets is assessed, and a normalized score between 1 and 5 is assigned.

    **Score Ranges:**
    - Entity Count: List-based sets (ASR and VTT extracted).
    - Final Score: Float between 1 (poor match) and 5 (strong match).

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    ASR Entities: {asr_entities}, VTT Entities: {vtt_entities}, Final Score: {entity_accuracy_score}.  
    Identify entity types frequently omitted or altered in subtitles.  
    Recommend alignment adjustments to ensure all key entities are retained.  
    Ensure response is formal, direct, and optimized for instructor review.
    """
    return learner_prompt, instructor_prompt

def prompt_entity_accuracy_batch(asr_entities_list, vtt_entities_list, entity_accuracy_scores):
    learner_prompt = f"""
    You are simulating a learner's perception of content reliability based on subtitle accuracy across a **batch of educational audios**. The **Entity Accuracy** metric reflects how well key names, terms, and references from the original spoken content are preserved in subtitles.

    **Evaluation Methodology:**
    - **ASR Entities / VTT Entities**: Sets of named entities (e.g., persons, places, organizations) extracted from audio and subtitles. Greater overlap suggests higher accuracy and better clarity of key content.
    - **Final Ratings**: Entity match scores per audio, scaled from 1 (poor match) to 5 (excellent alignment).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence batch-level summary.
    - Focus: Perceived consistency, clarity, and trustworthiness of named information across the entire audio set.

    **Prompt Instruction:**
    Given the following data across the batch:
    - ASR Entities List: {asr_entities_list}
    - VTT Entities List: {vtt_entities_list}
    - Entity Accuracy Ratings: {entity_accuracy_scores}

    Write a short learner-style reflection on the **overall perceived reliability** of subtitles with regard to names, terms, and references. Avoid technical detail. Emphasize whether the learner would have consistently felt confident in the correctness of key information.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch of educational audio samples** using the **Entity Accuracy** metric. This metric reflects how well named entities from the spoken transcript (ASR) are captured in the subtitle text (VTT).
    
    **Evaluation Methodology:**
    - **ASR Entities / VTT Entities**: Sets of named entities (e.g., persons, places, organizations) extracted from audio and subtitles. Greater overlap suggests higher accuracy and better clarity of key content.
    - **Final Ratings**: Entity match scores per audio, scaled from 1 (poor match) to 5 (excellent alignment).

    **Metric Overview:**
    - ASR Entities List: {asr_entities_list}
    - VTT Entities List: {vtt_entities_list}
    - Final Ratings (1–5): {entity_accuracy_scores}

    Analyze the batch and provide a concise **summary of entity alignment quality**.  
    Identify common patterns of omission or alteration across the set (e.g., frequent exclusion of organization names or technical terms).  
    Use passive voice, 2–4 lines, and provide focused recommendations for improving subtitle entity retention.  
    Avoid per-audio granularity — the feedback must reflect **overall trends** and alignment quality.
    """
    return learner_prompt, instructor_prompt

def prompt_entity_accuracy_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner's perception of content reliability based on subtitle accuracy across a **course** comprising multiple modules. The **Entity Accuracy** metric reflects how well names, terms, and references from the original spoken content are preserved in subtitles.

    **Evaluation Methodology:**
    - **ASR Entities / VTT Entities**: Named entities (e.g., persons, places, organizations) extracted from audio (ASR) and subtitle (VTT) transcripts. Higher overlap means clearer and more trustworthy key content.
    - **Final Ratings**: Per-module accuracy ratings, scaled from 1 (poor match) to 5 (excellent alignment).

    **Expected Response Guidelines:**
    - Perspective: Learner's point of view (not technical/AI).
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice wherever natural.
    - Focus: Overall clarity, trust, and consistency of key terms across the course.

    **Prompt Instruction:**
    Given the following **module-level data**, reflect on the **overall course-level experience** in terms of how reliably the subtitles conveyed the important named information:

    {module_level_data}

    Write a short learner-style reflection on the **perceived reliability and clarity** of subtitles across the course. Avoid technical jargon or per-module commentary. Focus on the learner’s confidence in the accuracy of key names and references.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Entity Accuracy** across a course composed of several modules. This metric assesses how effectively named entities from the audio (ASR) are retained in the subtitle (VTT) text.

    **Evaluation Methodology:**
    - **ASR Entities / VTT Entities**: Named entities such as persons, places, or organizations. High overlap indicates higher subtitle fidelity.
    - **Final Ratings**: Per-module scores, scaled from 1 (low accuracy) to 5 (high accuracy).

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a 2–4 line instructor-style summary of **overall entity alignment** quality across the course.
    - Identify general trends — e.g., common types of entities often omitted or misrepresented.
    - Offer a concise, actionable suggestion to improve subtitle accuracy without focusing on specific modules.

    Provide a high-level summary of subtitle fidelity in terms of entity retention throughout the course.
    """
    return learner_prompt, instructor_prompt

def parse_vtt(vtt_path, service_account_file):
    service = get_gdrive_service(service_account_file)
    content = None
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            request = service.files().get_media(fileId=vtt_path)
            file_content = request.execute()
            if isinstance(file_content, bytes):
                content = file_content.decode('utf-8')
            else:
                content = file_content
            break
        except (HttpError, http.client.IncompleteRead) as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to read VTT file {vtt_path} after {max_retries} attempts: {str(e)}")
                return []
            wait_time = 2 ** retries
            print(f"Attempt {retries} failed reading VTT file, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    if not content:
        print("No content found in VTT file.")
        return []

    entries = []
    blocks = re.split(r"\n\s*\n", content.strip())

    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) >= 2 and "-->" in lines[0]:
            time_line, *text_lines = lines
        elif len(lines) >= 3 and "-->" in lines[1]:
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            continue
        start, end = [t.strip() for t in time_line.split("-->")]
        text = " ".join(text_lines).strip()
        entries.append({"start": start, "end": end, "text": text})

    return entries

def vtt_time_to_seconds(ts):
    parts = re.split('[:.]', ts)
    if len(parts) == 4:  # HH:MM:SS.mmm
        h, m, s, ms = parts
    elif len(parts) == 3:  # MM:SS.mmm
        h = 0
        m, s, ms = parts
    else:
        raise ValueError(f"Unexpected VTT timestamp format: {ts}")
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    return seconds

def transcribe_audio(audio_path, service_account_file):
    global _whisper_model
    service = get_gdrive_service(service_account_file)

    max_retries = 5
    retries = 0
    wav_data = None

    while retries < max_retries:
        try:
            request = service.files().get_media(fileId=audio_path)
            wav_data = request.execute()
            break
        except (HttpError, http.client.IncompleteRead) as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to read WAV file {audio_path} after {max_retries} attempts: {str(e)}")
                return "", [], []
            wait_time = 2 ** retries
            print(f"Attempt {retries} failed reading WAV file, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    if not wav_data:
        print("No audio data found.")
        return "", [], []

    temp_dir = tempfile.gettempdir()
    temp_wav_path = os.path.join(temp_dir, f"temp_audio_{audio_path}.wav")

    try:
        with open(temp_wav_path, 'wb') as f:
            f.write(wav_data)

        device = get_device()
        compute_type = "float16" if device == "cuda" else "int8"
        if _whisper_model is None:
            _whisper_model = WhisperModel("base", device=device, compute_type=compute_type)

        segments, _ = _whisper_model.transcribe(
            temp_wav_path,
            word_timestamps=True,
            beam_size=1,
            vad_filter=False
        )

        segments_list = list(segments)
        transcript = " ".join(seg.text for seg in segments_list)

        words = []
        for segment in segments_list:
            if segment.words:
                words.extend([{"word": w.word, "start": w.start, "end": w.end} for w in segment.words])
            else:
                words.append({"word": segment.text, "start": segment.start, "end": segment.end})

        return transcript.strip(), words, segments_list

    except Exception as e:
        print(f"Error transcribing audio {audio_path}: {str(e)}")
        return "", [], []

    finally:
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_wav_path}: {str(e)}")

def extract_entities(text):
    global _nlp_model
    if _nlp_model is None:
        if get_device() == "cuda":
            spacy.require_gpu()
        _nlp_model = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    doc = _nlp_model(text)
    return set(ent.text for ent in doc.ents)

def calc_similarity_metrics(reference, candidate):
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    bleu = sentence_bleu([reference.split()], candidate.split())
    rouge_score_val = _rouge_scorer.score(reference, candidate)['rouge1'].fmeasure

    device = get_device()
    P, R, F1 = bert_score(
        [candidate],
        [reference],
        lang="en",
        verbose=False,
        batch_size=8 if device == "cuda" else 1,
        nthreads=4,
        rescale_with_baseline=False,
        device=device  # Ensure BERTScore uses GPU if available
    )
    bert = float(F1[0])
    return bleu, rouge_score_val, bert

def evaluate_subtitles(audio_path, vtt_path, service_account_file):
    # Parse VTT and transcribe audio in parallel if possible
    file_name = get_parent_folder_name(audio_path, service_account_file)
    vtt_entries = parse_vtt(vtt_path, service_account_file)
    vtt_text = " ".join([e["text"] for e in vtt_entries])
    asr_text, asr_words, asr_segments = transcribe_audio(audio_path, service_account_file)

    # Pre-compute timestamp conversions all at once
    vtt_times = [(vtt_time_to_seconds(e["start"]), vtt_time_to_seconds(e["end"])) for e in vtt_entries]
    asr_times = [(seg.start, seg.end) for seg in asr_segments]

    # 1. TIMING ALIGNMENT - How well subtitle timings match audio
    if len(vtt_times) == 0 or len(asr_times) == 0:
        return 0.0

    vtt_starts = np.array([t[0] for t in vtt_times])
    vtt_ends = np.array([t[1] for t in vtt_times])
    asr_starts = np.array([t[0] for t in asr_times])
    asr_ends = np.array([t[1] for t in asr_times])

    # Match each VTT start to the closest ASR start segment index
    matched_asr_indices = []
    for v_start in vtt_starts:
        diffs = np.abs(asr_starts - v_start)
        min_idx = np.argmin(diffs)
        matched_asr_indices.append(min_idx)

    matched_asr_starts = asr_starts[matched_asr_indices]
    matched_asr_ends = asr_ends[matched_asr_indices]

    start_diffs = np.abs(vtt_starts - matched_asr_starts)
    end_diffs = np.abs(vtt_ends - matched_asr_ends)

    max_acceptable_diff = 0.5  # 0.2 seconds
    decay_factor = 1.0  # Controls how quickly similarity decreases with difference

    start_sim = np.exp(-decay_factor * start_diffs)
    end_sim = np.exp(-decay_factor * end_diffs)
    segment_similarities = (start_sim + end_sim) / 2

    timing_alignment = float(np.mean(segment_similarities))

    # 2. SEMANTIC MATCH - How well subtitle text matches audio content semantically
    bleu, rouge, bert = calc_similarity_metrics(asr_text, vtt_text)
    # Normalize scores (BERT is usually in [0, 1], BLEU & ROUGE sometimes out of scale)
    bleu_norm = min(max(bleu, 0), 1)
    rouge_norm = min(max(rouge, 0), 1)
    bert_norm = min(max(bert, 0), 1)
    # Weighted fusion — you can tune these weights based on experimentation
    semantic_match = round(0.5 * bert_norm + 0.3 * rouge_norm + 0.2 * bleu_norm, 4)

    # 3. CONTENT COVERAGE - How completely the subtitles cover the audio content
    asr_words_set = set(asr_text.lower().split())
    vtt_words_set = set(vtt_text.lower().split())
    missing = len(asr_words_set - vtt_words_set) / max(1, len(asr_words_set))
    extra = len(vtt_words_set - asr_words_set) / max(1, len(vtt_words_set))

    # Convert to scores where 1 is good (no missing/extra words)
    content_coverage = (1 - missing + 1 - extra) / 2
    content_coverage = max(0.0, min(1.0, content_coverage))

    # 4. ENTITY ACCURACY - How well key entities in audio match with subtitles
    asr_ents = extract_entities(asr_text)
    vtt_ents = extract_entities(vtt_text)
    if len(asr_ents | vtt_ents) > 0:
        entity_accuracy = 1 - (len(asr_ents.symmetric_difference(vtt_ents)) / len(asr_ents | vtt_ents))
    else:
        entity_accuracy = 1.0  # If no entities in either, consider it perfect match

    timing_alignment = scale_to_rating(timing_alignment)
    semantic_match = scale_to_rating(semantic_match)
    content_coverage = scale_to_rating(content_coverage)
    entity_accuracy = scale_to_rating(entity_accuracy)

    timing_alignment_user_feedback = execute_prompt(prompt_timing_alignment(start_diffs, end_diffs, max_acceptable_diff, timing_alignment)[0])
    timing_alignment_instructor_feedback = execute_prompt(prompt_timing_alignment(start_diffs, end_diffs, max_acceptable_diff, timing_alignment)[1])
    semantic_match_user_feedback = execute_prompt(prompt_semantic_match(bleu_norm, rouge_norm, bert_norm, semantic_match)[0])
    semantic_match_instructor_feedback = execute_prompt(prompt_semantic_match(bleu_norm, rouge_norm, bert_norm, semantic_match)[1])
    content_coverage_user_feedback = execute_prompt(prompt_content_coverage(missing, extra, content_coverage)[0])
    content_coverage_instructor_feedback = execute_prompt(prompt_content_coverage(missing, extra, content_coverage)[1])
    entity_accuracy_user_feedback = execute_prompt(prompt_entity_accuracy(asr_ents, vtt_ents, entity_accuracy)[0])
    entity_accuracy_instructor_feedback = execute_prompt(prompt_entity_accuracy(asr_ents, vtt_ents, entity_accuracy)[1])

    results = {
        "File": file_name,
        "Timing Alignment Score": {"Intermediate Parameters":{"Start Diffs": start_diffs, "End Diffs": end_diffs, "Max Acceptable Diff": max_acceptable_diff}, "Final Score": timing_alignment, "Learner Perspective Assessment": timing_alignment_user_feedback, "Instructor Feedback": timing_alignment_instructor_feedback},
        "Semantic Match Score": {"Intermediate Parameters":{"BLEU Norm": bleu_norm, "BERT Norm": bert_norm, "ROUGE Norm": rouge_norm}, "Final Score": semantic_match, "Learner Perspective Assessment": semantic_match_user_feedback, "Instructor Feedback": semantic_match_instructor_feedback},
        "Content Coverage Score": {"Intermediate Parameters":{"Missing": missing, "Extra": extra}, "Final Score": content_coverage, "Learner Perspective Assessment": content_coverage_user_feedback, "Instructor Feedback": content_coverage_instructor_feedback},
        "Entity Accuracy Score": {"Intermediate Parameters":{"Audio Entities": asr_ents, "VTT Entities": vtt_ents}, "Final Score": entity_accuracy, "Learner Perspective Assessment": entity_accuracy_user_feedback, "Instructor Feedback": entity_accuracy_instructor_feedback},
        }

    return results

class AudioAnalysisDatasetv2(Dataset):
    def __init__(self, audio_files, vtt_files):
        self.audio_files = audio_files
        self.vtt_files = vtt_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], self.vtt_files[idx]

def analyze_batch_2(batch, service_account_file):
    results = []
    for audio_path, vtt_path in batch:
        try:
            result = evaluate_subtitles(audio_path, vtt_path, service_account_file)
            results.append(result)
        except Exception as e:
            print(f"Failed to process {audio_path}: {str(e)}")
            results.append({"error": str(e)})
    return results

def perform_audio_subtitle_analysis(service_account_file, folder_id):
    result = {}
    service = get_gdrive_service(service_account_file)
    result["Course Name"] = get_folder_name(folder_id, service_account_file)
    # Try to fetch metadata.json
    metadata = fetch_metadata_json(service, folder_id)
    extensions = ['.wav', '.vtt']
    all_files = find_files_recursively(service, folder_id, extensions)
    module_files = organize_files_by_module(metadata, all_files)
    audio_subtitle_summaries = []
    total_results = []
    count = 0
    for key, value in metadata.items():
        if count >= 2:
          break
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            audio_files = [d["wav_file"] for d in module_files[module_name] if "wav_file" in d][:2]
            vtt_files = [d["vtt_file"] for d in module_files[module_name] if "vtt_file" in d][:2]
            dataset = AudioAnalysisDatasetv2(audio_files, vtt_files)
            loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x, num_workers=os.cpu_count()-1)
            module_results = {}
            module_results["Name"] = module_name
            module_results["Results"] = []
            for batch in loader:
                temp_ = analyze_batch_2(batch, service_account_file)
                audio_subtitle_summaries.extend(temp_)
                module_results["Results"].extend(temp_)
            module_results["Timing Alignment Score"] = {}
            timing_alignment_ratings = [each["Timing Alignment Score"]["Final Score"] for each in module_results["Results"]]
            start_diffs_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["Start Diffs"] for each in module_results["Results"]]
            end_diffs_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["End Diffs"] for each in module_results["Results"]]
            max_acceptable_diff_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["Max Acceptable Diff"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_timing_alignment_batch(start_diffs_list, end_diffs_list, max_acceptable_diff_list, timing_alignment_ratings)
            module_results["Timing Alignment Score"]["Score"] = round(sum(timing_alignment_ratings)/len(timing_alignment_ratings), 2)
            module_results["Timing Alignment Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Timing Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results["Semantic Match Score"] = {}
            semantic_match_ratings = [each["Semantic Match Score"]["Final Score"] for each in module_results["Results"]]
            bleu_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["BLEU Norm"] for each in module_results["Results"]]
            bert_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["BERT Norm"] for each in module_results["Results"]]
            rouge_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["ROUGE Norm"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_semantic_match_batch(bleu_norm_list, bert_norm_list, rouge_norm_list, semantic_match_ratings)
            module_results["Semantic Match Score"]["Score"] = round(sum(semantic_match_ratings)/len(semantic_match_ratings), 2)
            module_results["Semantic Match Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Semantic Match Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results["Content Coverage Score"] = {}
            content_coverage_ratings = [each["Content Coverage Score"]["Final Score"] for each in module_results["Results"]]
            missing_list = [each["Content Coverage Score"]["Intermediate Parameters"]["Missing"] for each in module_results["Results"]]
            extra_list = [each["Content Coverage Score"]["Intermediate Parameters"]["Extra"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_content_coverage_batch(missing_list, extra_list, content_coverage_ratings)
            module_results["Content Coverage Score"]["Score"] = round(sum(content_coverage_ratings)/len(content_coverage_ratings), 2)
            module_results["Content Coverage Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Content Coverage Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results["Entity Accuracy Score"] = {}
            entity_accuracy_ratings = [each["Entity Accuracy Score"]["Final Score"] for each in module_results["Results"]]
            audio_entities_list = [each["Entity Accuracy Score"]["Intermediate Parameters"]["Audio Entities"] for each in module_results["Results"]]
            vtt_entities_list = [each["Entity Accuracy Score"]["Intermediate Parameters"]["VTT Entities"] for each in module_results["Results"]]
            learner_prompt, instructor_prompt = prompt_entity_accuracy_batch(audio_entities_list, vtt_entities_list, entity_accuracy_ratings)
            module_results["Entity Accuracy Score"]["Score"] = round(sum(entity_accuracy_ratings)/len(entity_accuracy_ratings), 2)
            module_results["Entity Accuracy Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results["Entity Accuracy Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            total_results.append(module_results)
            count += 1
    result["Module Results"] = total_results
    result["Course Results"] = {}
    result["Course Results"]["Timing Alignment Score"] = {}
    result["Course Results"]["Semantic Match Score"] = {}
    result["Course Results"]["Content Coverage Score"] = {}
    result["Course Results"]["Entity Accuracy Score"] = {}
    timing_alignment_parameter = ""
    semantic_match_parameter = ""
    content_coverage_parameter = ""
    entity_accuracy_parameter = ""
    final_timing_alignment_scores = []
    final_semantic_match_scores = []
    final_content_coverage_scores = []
    final_entity_accuracy_scores = []
    for module_result in result["Module Results"]:
        module_name = module_result["Name"]
        timing_alignment_ratings = [each["Timing Alignment Score"]["Final Score"] for each in module_result["Results"]]
        start_diffs_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["Start Diffs"] for each in module_result["Results"]]
        end_diffs_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["End Diffs"] for each in module_result["Results"]]
        max_acceptable_diff_list = [each["Timing Alignment Score"]["Intermediate Parameters"]["Max Acceptable Diff"] for each in module_result["Results"]]
        timing_alignment_parameter += (
            f"Module Name: {module_name}\n"
            f"Start Time Difference (s): {start_diffs_list}\n"
            f"End Time Difference (s): {end_diffs_list}\n"
            f"Max Acceptable Diff (s): {max_acceptable_diff_list}\n"
            f"Final Ratings: {timing_alignment_ratings}\n"
        )
        final_timing_alignment_scores.append(module_result["Timing Alignment Score"]["Score"])
        semantic_match_ratings = [each["Semantic Match Score"]["Final Score"] for each in module_result["Results"]]
        bleu_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["BLEU Norm"] for each in module_result["Results"]]
        bert_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["BERT Norm"] for each in module_result["Results"]]
        rouge_norm_list = [each["Semantic Match Score"]["Intermediate Parameters"]["ROUGE Norm"] for each in module_result["Results"]]
        semantic_match_parameter += (
            f"Module Name: {module_name}\n"
            f"BLEU Scores: {bleu_norm_list}\n"
            f"BERT Scores: {bert_norm_list}\n"
            f"ROUGE Scores: {rouge_norm_list}\n"
            f"Final Semantic Match Ratings: {semantic_match_ratings}\n"
        )
        final_semantic_match_scores.append(module_result["Semantic Match Score"]["Score"])
        content_coverage_ratings = [each["Content Coverage Score"]["Final Score"] for each in module_result["Results"]]
        missing_list = [each["Content Coverage Score"]["Intermediate Parameters"]["Missing"] for each in module_result["Results"]]
        extra_list = [each["Content Coverage Score"]["Intermediate Parameters"]["Extra"] for each in module_result["Results"]]
        content_coverage_parameter += (
            f"Module Name: {module_name}\n"
            f"Missing Ratios: {missing_list}\n"
            f"Extra Ratios: {extra_list}\n"
            f"Final Content Coverage Ratings: {content_coverage_ratings}\n"
        )
        final_content_coverage_scores.append(module_result["Content Coverage Score"]["Score"])
        entity_accuracy_ratings = [each["Entity Accuracy Score"]["Final Score"] for each in module_result["Results"]]
        audio_entities_list = [each["Entity Accuracy Score"]["Intermediate Parameters"]["Audio Entities"] for each in module_result["Results"]]
        vtt_entities_list = [each["Entity Accuracy Score"]["Intermediate Parameters"]["VTT Entities"] for each in module_result["Results"]]
        entity_accuracy_parameter += (
            f"Module Name: {module_name}\n"
            f"ASR Entities List: {audio_entities_list}\n"
            f"VTT Entities List: {vtt_entities_list}\n"
            f"Final Entity Accuracy Ratings: {entity_accuracy_ratings}\n"
        )
        final_entity_accuracy_scores.append(module_result["Entity Accuracy Score"]["Score"])  
    learner_prompt, instructor_prompt = prompt_timing_alignment_course(timing_alignment_parameter)
    result["Course Results"]["Timing Alignment Score"]["Score"] = round(sum(final_timing_alignment_scores)/len(final_timing_alignment_scores),2)
    result["Course Results"]["Timing Alignment Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Timing Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_semantic_match_course(semantic_match_parameter)
    result["Course Results"]["Semantic Match Score"]["Score"] = round(sum(final_semantic_match_scores)/len(final_semantic_match_scores),2)
    result["Course Results"]["Semantic Match Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Semantic Match Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_content_coverage_course(content_coverage_parameter)
    result["Course Results"]["Content Coverage Score"]["Score"] = round(sum(final_content_coverage_scores)/len(final_content_coverage_scores),2)
    result["Course Results"]["Content Coverage Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Content Coverage Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_entity_accuracy_course(entity_accuracy_parameter)
    result["Course Results"]["Entity Accuracy Score"]["Score"] = round(sum(final_entity_accuracy_scores)/len(final_entity_accuracy_scores),2)
    result["Course Results"]["Entity Accuracy Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result["Course Results"]["Entity Accuracy Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    save_dict_to_json(convert_to_serializable(result), "Audio Subtitle Validation Results.json")

if __name__ == "__main__":
    service_account_file = ""
    folder_id = ""
    start_time = time.time()
    perform_audio_analysis(service_account_file, folder_id)
    perform_audio_subtitle_analysis(service_account_file, folder_id)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Time taken: {elapsed_minutes:.2f} minutes")
