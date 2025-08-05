import os
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import time
import spacy
from transformers import pipeline
import google.generativeai as genai
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
import http.client
import faiss
import pyphen
import tempfile
import base64
from googleapiclient.http import MediaIoBaseDownload
from google.api_core.exceptions import InternalServerError, ResourceExhausted

# Initialize GPU acceleration
if torch.cuda.is_available():
    spacy.prefer_gpu()
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("No GPU available, using CPU")

nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")
dictionary = pyphen.Pyphen(lang='en')

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Initialize the model
print("Loading sentence transformer model...")
t_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
print("Model loaded successfully!")

def get_gdrive_service(service_account_file):
    """Create and return a Google Drive service using service account credentials."""
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def read_transcript(file_id, service_account_file, max_retries=5):
    """Read and clean a transcript from a file with retry logic"""
    service = get_gdrive_service(service_account_file)
    retries = 0
    while retries < max_retries:
        try:
            # Get the file content
            request = service.files().get_media(fileId=file_id)
            file_content = request.execute()

            # Convert bytes to string if necessary
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')

            content = file_content.strip()
            # Clean the transcript text
            content = re.sub(r'\n+', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            return content
        except (HttpError, http.client.IncompleteRead) as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to read file {file_id} after {max_retries} attempts: {str(e)}")
                return ""
            # Exponential backoff
            wait_time = 2 ** retries
            print(f"Attempt {retries} failed, retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def download_pdf_temporarily(drive_service, file_id):
    """Download PDF file from Google Drive to a temporary file and return the path"""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_file_path = temp_file.name

    try:
        # Download from Google Drive
        request = drive_service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(temp_file, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}%")

        temp_file.close()
        print(f"PDF temporarily downloaded to: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        # Clean up on error
        temp_file.close()
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise e

def get_pdf_content_with_gemini(file_path: str, custom_prompt: str = None) -> str:
    """
    Gets comprehensive content extraction from a PDF file using Gemini.
    This will include text content and insights from images.
    """
    if custom_prompt is None:
        prompt = (
            "Analyze the attached PDF document comprehensively and extract all content including:\n"
            "1. All readable text content, maintaining structure and formatting\n"
            "2. Detailed descriptions of all images, charts, diagrams, tables, and graphs\n"
            "3. Document structure including headers, sections, bullet points\n"
            "4. Key information such as data points, numbers, dates, names\n"
            "5. Context and understanding of the document's purpose and main themes\n\n"
            "Please organize the extracted content in a clear, structured format that preserves "
            "the original document's meaning and organization. Include both textual content and "
            "visual element descriptions."
        )
    else:
        prompt = custom_prompt

    try:
        # Read file and encode to base64
        with open(file_path, 'rb') as file:
            file_bytes = file.read()

        file_part = {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": base64.b64encode(file_bytes).decode('utf-8')
                    }
                }
            ]
        }

        # Use the vision model for processing multimodal content
        model_for_vision = model
        response = model_for_vision.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": prompt}]},
                file_part
            ]
        )
        return response.text

    except Exception as e:
        print("Error in gemini processing of pdf file")
        return ""  # Return empty string on failure

def get_file_name_pdf(drive_file_id, service_account_file):
    service = get_gdrive_service(service_account_file)
    file = service.files().get(fileId=drive_file_id, fields='name').execute()
    return file.get('name')

def read_reading(drive_file_id, service_account_file, custom_prompt):
    """Main function to download PDF from Drive temporarily and extract content with Gemini"""
    temp_file_path = None
    file_name = get_file_name_pdf(drive_file_id, service_account_file)
    print(f"File Name: {file_name}")
    try:
        # Create Drive service
        print("Authenticating with Google Drive...")
        drive_service = get_gdrive_service(service_account_file)

        # Download PDF temporarily
        print(f"Downloading PDF with ID: {drive_file_id}")
        temp_file_path = download_pdf_temporarily(drive_service, drive_file_id)

        # Process with Gemini
        print("Processing PDF with Gemini...")
        extracted_content = get_pdf_content_with_gemini(
            file_path=temp_file_path,
            custom_prompt=custom_prompt
        )

        return (file_name, extracted_content)

    except Exception as e:
        print(f"Error processing PDF from Drive: {str(e)}")
        return (file_name, "")

    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                print(f"Error deleting temporary file: {e}")

def syllable_count(word):
    return len(dictionary.inserted(word).split("-"))

def calculate_easy_to_understand_score(doc):
    word_freq = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    word_length_avg = np.mean([len(word) for word in word_freq])
    syllables_avg = np.mean([syllable_count(word) for word in word_freq])
    raw_score = max(0, 16 - (0.6 * word_length_avg + 0.3 * syllables_avg))
    score = round(float(np.clip(raw_score / 3, 1, 5)), 2)

    return {
        'score': score,
        'intermediates': {
            'word_length_avg': round(word_length_avg, 2),
            'syllables_avg': round(syllables_avg, 2),
            'raw_score': round(raw_score, 2)
        }
    }

def classify_engagement_feel_dynamic(sentences):
    sentiments = sentiment_analyzer(sentences)
    sentiment_scores = [s['score'] if s['label'] == 'POSITIVE' else -s['score'] for s in sentiments]
    avg_sentiment = np.mean(sentiment_scores)
    
    # Hyperparameters tweaked for higher scores:
    raw_score = avg_sentiment * 12  # Increase the spread a bit (was 10)
    # Offset increased (+11) and divisor reduced (now / 3.5, was 4)
    score = round(float(np.clip((raw_score + 11) / 3.5, 1, 5)), 2)

    return {
        'score': score,
        'intermediates': {
            'avg_sentiment': round(avg_sentiment, 3),
            'positive_ratio': round(sum(1 for s in sentiments if s['label'] == 'POSITIVE') / len(sentiments), 3),
            'raw_score': round(raw_score, 2)
        }
    }

def is_course_well_structured_score(sentences, embeddings):
    n_clusters = min(5, len(sentences))
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, n_clusters, niter=10, verbose=False, seed=42)
    kmeans.train(embeddings.cpu().numpy())
    cluster_labels = kmeans.index.search(embeddings.cpu().numpy(), 1)[1].flatten()
    unique_clusters = len(set(cluster_labels))
    cluster_diversity = unique_clusters / n_clusters
    raw_score = cluster_diversity * 10
    score = round(float(np.clip(raw_score / 2, 1, 5)), 2)

    return {
        'score': score,
        'intermediates': {
            'unique_clusters': unique_clusters,
            'total_clusters': n_clusters,
            'cluster_diversity': round(cluster_diversity, 3),
            'raw_score': round(raw_score, 2)
        }
    }

def pacing_and_flow_score(sentences, embeddings):
    similarities = util.cos_sim(embeddings[:-1], embeddings[1:]).diagonal().tolist()
    avg_similarity = np.mean(similarities)
    similarity_std = np.std(similarities)
    raw_score = avg_similarity * 12 + 0.8  # Was 10
    score = round(float(np.clip(raw_score / 2.4, 1, 5)), 2)  # Was /2

    return {
        'score': score,
        'intermediates': {
            'avg_similarity': round(avg_similarity, 3),
            'similarity_std': round(similarity_std, 3),
            'consistency_score': round(1 - similarity_std, 3),
            'raw_score': round(raw_score, 2)
        }
    }

def learning_value_score(sentences, embeddings):
    centroid = torch.mean(embeddings, dim=0, keepdim=True)
    similarities_to_centroid = util.cos_sim(embeddings, centroid).squeeze().cpu().numpy()
    avg_coherence = np.mean(similarities_to_centroid)
    coherence_std = np.std(similarities_to_centroid)

    # Higher multiplier and small positive offset for higher raw scores
    raw_score = avg_coherence * 12 + 0.7  # was: avg_coherence * 10
    score = round(float(np.clip(raw_score / 2.5, 1, 5)), 2)  # was: /2

    return {
        'score': score,
        'intermediates': {
            'avg_coherence': round(avg_coherence, 3),
            'coherence_std': round(coherence_std, 3),
            'focus_consistency': round(1 - coherence_std, 3),
            'raw_score': round(raw_score, 2)
        }
    }

def prompt_easy_to_understand(word_length_avg, syllables_avg, raw_score, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how easy the course language is to read and understand. The **Easy-to-Understand** metric captures how approachable and accessible the course content feels in terms of language complexity.

    **Evaluation Methodology:**
    - **Average Word Length:** Reflects vocabulary complexity; shorter words suggest simpler language.
    - **Average Syllables per Word:** Indicates pronunciation effort; fewer syllables typically improve readability.
    - **Raw Complexity Score (0–10):** Synthesizes the above to assess linguistic simplicity.
    - **Final Rating:** Scaled from 1 (difficult, dense language) to 5 (very easy to follow).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: How understandable the content felt while reading or listening.

    **Prompt Instruction:**
    Given Word Length Avg: {word_length_avg} characters, Syllables Avg: {syllables_avg}, Raw Score: {raw_score}/10, and Final Score: {final_rating}, write a short learner-style reflection on content readability. Reflect how easily the material could be followed, with reference to language clarity and effort required.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the linguistic accessibility of educational content using the **Easy-to-Understand Score**, which is derived from word and syllable characteristics.

    **Methodology & Overview:**
    - Average Word Length: Shorter words enhance readability.
    - Average Syllables per Word: Lower syllables improve comprehension.
    - Raw Complexity Score: Aggregates vocabulary and phonetic complexity (range: 0–10).

    The final score is computed from these inputs and scaled between 1 (hard to follow) and 5 (highly accessible), with no manual capping.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Average Word Length: {word_length_avg} characters,  
    Average Syllables per Word: {syllables_avg},  
    Raw Complexity Score: {raw_score}/10,  
    Final Rating: {final_rating}/5.  

    Please provide a brief, 2–4 line feedback identifying which metric contributes most to reduced accessibility.  
    Suggestions should target that specific issue (e.g., shortening words or reducing syllables), stated in passive voice.  
    Avoid lengthy analysis — keep it formal, clear, and focused.
    """
    return learner_prompt, instructor_prompt

def prompt_easy_to_understand_batch(word_length_avg_list, syllables_avg_list, raw_score_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how easy the **language across a batch of educational content** is to read and comprehend. The **Easy-to-Understand** metric captures how approachable and accessible the material feels based on linguistic complexity.

    **Evaluation Methodology:**
    - **Average Word Length:** Shorter words typically suggest simpler vocabulary.
    - **Average Syllables per Word:** Fewer syllables tend to improve pronunciation ease and readability.
    - **Raw Complexity Score (0–10):** Synthesizes both metrics to quantify simplicity.
    - **Final Ratings:** Scaled from 1 (very difficult, verbose) to 5 (extremely easy to understand).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence batch-level reflection, mostly in passive voice.
    - Focus: General perception of clarity and effort required across the content set.

    **Prompt Instruction:**
    Given the following batch-level inputs:
    - Word Length Averages: {word_length_avg_list}
    - Syllables per Word: {syllables_avg_list}
    - Raw Complexity Scores: {raw_score_list}
    - Final Ratings: {final_rating_list}

    Write a learner-style reflection summarizing how easy or difficult the language **felt overall**. Avoid explaining the metrics. Instead, focus on how smooth or effortful it was to understand the content as a whole.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating a **batch of educational content** for linguistic accessibility using the **Easy-to-Understand Score**, based on word and syllable complexity.

    **Evaluation Methodology:**
    - **Average Word Length:** Shorter words typically suggest simpler vocabulary.
    - **Average Syllables per Word:** Fewer syllables tend to improve pronunciation ease and readability.
    - **Raw Complexity Score (0–10):** Synthesizes both metrics to quantify simplicity.
    - **Final Ratings:** Scaled from 1 (very difficult, verbose) to 5 (extremely easy to understand).

    **Metric Overview:**
    - Word Length Averages: {word_length_avg_list}
    - Syllables per Word: {syllables_avg_list}
    - Raw Complexity Scores (0–10): {raw_score_list}
    - Final Ratings (1–5): {final_rating_list}

    The final score reflects how approachable the content is to learners and is scaled from 1 (complex) to 5 (simple and accessible).

    **Instruction:**
    Analyze the metrics collectively. Identify which **dimension consistently reduces accessibility** across the set — word length or syllables.  
    Write a brief, formal, 2–4 line summary in passive voice, offering targeted improvement suggestions at the batch level. Avoid per-sample analysis or metric definitions — focus on **systemic trends**.
    """
    return learner_prompt, instructor_prompt

def prompt_easy_to_understand_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how easy the **language across a course** is to read and comprehend.  
    The **Easy-to-Understand** metric reflects how approachable and accessible the material feels based on overall linguistic complexity across all modules.

    **Evaluation Methodology:**
    - **Average Word Length:** Shorter words typically indicate simpler vocabulary.
    - **Average Syllables per Word:** Fewer syllables improve readability and pronunciation ease.
    - **Raw Complexity Score (0–10):** Combines both metrics to quantify simplicity.
    - **Final Ratings:** Scaled from 1 (very difficult) to 5 (extremely easy to understand).
    - **Alignment Balance Ratings:** Reflects fairness in coverage of learning objectives (included for broader learner perspective).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence reflection in mostly passive voice.
    - Focus: Overall clarity, effort required, and reading smoothness throughout the course.

    **Prompt Instruction:**
    Given the following **module-level data**:

    {module_level_data}

    Write a short learner-style reflection summarizing how easy or difficult the language felt **across the course**.  
    Avoid technical explanation of metrics. Focus instead on the perceived clarity, simplicity, and reading effort required by the learner.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **linguistic accessibility** of a course using the **Easy-to-Understand Score** and **Alignment Balance** for contextual insight.

    **Evaluation Methodology:**
    - **Average Word Length:** Shorter words generally mean simpler vocabulary.
    - **Average Syllables per Word:** Fewer syllables improve readability.
    - **Raw Complexity Score (0–10):** Combines both metrics to indicate simplicity.
    - **Final Ratings (1–5):** 1 = complex, 5 = simple and accessible.
    - **Alignment Balance Ratings:** Additional reference to ensure language accessibility does not undermine even coverage of learning objectives.

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a concise 2–4 line **course-level** evaluation.
    - Identify the dimension (word length or syllables) most consistently reducing accessibility.
    - Offer targeted, actionable improvement suggestions.
    - Avoid module-specific analysis.

    Provide a high-level summary of systemic linguistic trends affecting accessibility across the course.
    """
    
    return learner_prompt, instructor_prompt

def prompt_engagement(avg_sentiment, positive_ratio, raw_score, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s emotional response while interacting with the course content. The **Engagement Score** reflects how motivational, emotionally resonant, and inspiring the course feels based on language tone.

    **Evaluation Methodology:**
    - **Average Sentiment** (-1 to 1): Indicates overall positivity of the content.
    - **Positive Ratio** (0 to 1): Measures how much of the content carries a positive, uplifting tone.
    - **Raw Engagement Score** (-10 to 10): Synthesizes emotional and motivational cues.
    - **Final Rating:** Scaled from 1 (uninspiring) to 5 (highly engaging and motivating).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Emotional connection, motivation, and whether the content feels inspiring or flat.

    **Prompt Instruction:**
    Given Average Sentiment: {avg_sentiment}, Positive Ratio: {positive_ratio}, Raw Score: {raw_score}/10, and Final Rating: {final_rating}, write a short learner-style reflection on emotional engagement. Reflect how lively, uplifting, or monotonous the content felt throughout the course.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating instructional content based on the **Engagement Score**, derived from sentiment analysis indicators.

    **Methodology & Overview:**
    - Average Sentiment (Range: -1 to 1): Reflects affective tone of the instruction.
    - Positive Ratio (0 to 1): Measures the frequency of encouraging language.
    - Raw Engagement Score (Range: -10 to 10): Aggregates sentiment data into a single measure.
    - Final Score: Scaled from the above metrics to a 1 (low) to 5 (high) rating, where 5 signals strong learner engagement.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Average Sentiment: {avg_sentiment}, 
    Positive Ratio: {positive_ratio}, 
    Raw Score: {raw_score}/10, 
    Final Rating: {final_rating}.

    Please provide a 2–4 line summary identifying the metric farthest from ideal.  
    Offer targeted suggestions in passive voice to enhance learner engagement accordingly.  
    Keep the tone formal, factual, and concise.
    """
    return learner_prompt, instructor_prompt

def prompt_engagement_batch(avg_sentiment_list, positive_ratio_list, raw_score_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s emotional response while interacting with a **batch of educational content**. The **Engagement Score** reflects how motivational, emotionally resonant, and inspiring the course feels overall, based on its tone and delivery across multiple segments.

    **Evaluation Methodology:**
    - **Average Sentiment** (-1 to 1): Indicates overall emotional tone.
    - **Positive Ratio** (0 to 1): Measures frequency of uplifting and motivational language.
    - **Raw Engagement Score** (-10 to 10): Synthesizes emotional and motivational cues.
    - **Final Ratings:** Scaled from the above metrics to a 1–5 score, where 5 = highly engaging.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner experience across the batch.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence summary.
    - Focus: Emotional connection and consistency of motivational delivery.

    **Prompt Instruction:**
    Given the following batch-level input:
    - Average Sentiment values: {avg_sentiment_list}
    - Positive Ratio values: {positive_ratio_list}
    - Raw Engagement Scores: {raw_score_list}
    - Final Ratings: {final_rating_list}

    Write a short learner-style reflection summarizing the **overall emotional engagement** experienced across the content. Avoid metric explanation. Focus on how the course *felt* in terms of motivation and emotional resonance across the board.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **engagement quality** of a batch of educational materials using sentiment-derived metrics.

    **Evaluation Methodology:**
    - **Average Sentiment** (-1 to 1): Indicates overall emotional tone.
    - **Positive Ratio** (0 to 1): Measures frequency of uplifting and motivational language.
    - **Raw Engagement Score** (-10 to 10): Synthesizes emotional and motivational cues.
    - **Final Ratings:** Scaled from the above metrics to a 1–5 score, where 5 = highly engaging.

    **Batch-Level Inputs:**
    - Average Sentiment (-1 to 1): {avg_sentiment_list}
    - Positive Ratio (0 to 1): {positive_ratio_list}
    - Raw Engagement Scores (-10 to 10): {raw_score_list}
    - Final Ratings (1–5): {final_rating_list}

    **Instruction:**
    Review these batch-level scores and write a concise, formal, and improvement-oriented summary (2–4 lines).  
    Identify the metric with the **most consistent deviation from ideal**, and offer a brief recommendation (in passive voice) on how engagement could be enhanced.  
    Avoid sample-specific feedback — focus on the macro-level learner experience and improvement path.
    """
    return learner_prompt, instructor_prompt

def prompt_engagement_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s emotional response while experiencing a **course** composed of multiple modules.  
    The **Engagement Score** reflects how motivational, emotionally resonant, and inspiring the course feels overall, based on tone and delivery throughout.

    **Evaluation Methodology:**
    - **Average Sentiment** (-1 to 1): Indicates overall emotional tone.
    - **Positive Ratio** (0 to 1): Measures frequency of uplifting and motivational language.
    - **Raw Engagement Score** (-10 to 10): Synthesizes emotional and motivational cues.
    - **Final Ratings**: Scaled from the above metrics to a 1–5 score, where 5 = highly engaging.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner experience across the entire course.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary using passive voice wherever natural.
    - Focus: Emotional connection, motivation, and consistency of delivery.

    **Prompt Instruction:**
    Given the following **module-level engagement data**, reflect on the **overall course-level engagement experience**:

    {module_level_data}

    Write a short learner-style reflection summarizing how engaging and motivational the course felt overall.  
    Avoid technical explanation of metrics — focus on the learner’s perceived emotional resonance and consistency of motivation.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **engagement quality** of a course composed of multiple modules, using sentiment-derived metrics.

    **Evaluation Methodology:**
    - **Average Sentiment** (-1 to 1): Indicates overall emotional tone.
    - **Positive Ratio** (0 to 1): Measures frequency of uplifting and motivational language.
    - **Raw Engagement Score** (-10 to 10): Synthesizes emotional and motivational cues.
    - **Final Ratings**: Scaled from the above metrics to a 1–5 score, where 5 = highly engaging.

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Provide a 2–4 line, improvement-oriented summary of overall course engagement quality.
    - Identify the metric that most consistently deviated from ideal performance.
    - Suggest one concise, actionable improvement for enhancing emotional engagement across the course.
    - Avoid module-specific detail; focus on overall trends.

    Provide a high-level summary reflecting the macro-level learner engagement experience and improvement path.
    """
    
    return learner_prompt, instructor_prompt

def prompt_pacing_flow(avg_similarity, similarity_std, consistency_score, raw_score, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how smoothly and clearly the lesson progressed. The **Pacing & Flow** metric reflects the logical progression and rhythm of the course content.

    **Evaluation Methodology:**
    - **Average Similarity (-1 to 1):** Measures how well consecutive sections connect; higher values reflect smoother transitions.
    - **Similarity Standard Deviation (≥0):** Captures flow stability; lower values indicate consistent pacing.
    - **Consistency Score (0 to 1):** Derived from deviation, indicating steadiness of structure.
    - **Raw Flow Score (-10 to 10):** Aggregates the above into a directional measure of overall flow.
    - **Final Rating (1–5):** Higher scores represent smoother, more cohesive delivery.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: How naturally the content progressed and whether it felt abrupt, choppy, or steady.

    **Prompt Instruction:**
    Given Avg Similarity: {avg_similarity}, Similarity Std Dev: {similarity_std}, Consistency Score: {consistency_score}, Raw Score: {raw_score}/10, and Final Rating: {final_rating}, write a short learner-style reflection on the pacing and flow. Avoid technical terms. Reflect whether the content felt well-paced and easy to follow.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating instructional coherence using the **Pacing & Flow Score**, which reflects thematic progression, consistency, and structural clarity.

    **Methodology & Overview:**
    - **Average Similarity** (−1 to 1): Indicates sentence-level thematic continuity.
    - **Similarity Standard Deviation** (≥0): Lower values imply more predictable pacing.
    - **Consistency Score** (0 to 1): Evaluates transition uniformity.
    - **Raw Flow Score** (−10 to 10): Captures directional structure.

    **Final Score Computation:**
    Intermediate metrics are aggregated and normalized into a final score from 1 (poor flow) to 5 (excellent flow).

    **Feedback (passive voice, 2–4 lines, improvement-focused):**
    Average Similarity: {avg_similarity}
    Similarity Std Dev: {similarity_std}
    Consistency Score: {consistency_score}
    Raw Flow Score: {raw_score}/10  
    Final Rating: {final_rating}/5  

    Based on the metric deviating most from the ideal, specific pacing improvements are recommended.  
    Focus should be placed on enhancing sentence continuity, reducing variability, or smoothing transitions accordingly.
    """
    return learner_prompt, instructor_prompt

def prompt_pacing_flow_batch(avg_similarity_list, similarity_std_list, consistency_score_list, raw_score_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how smoothly and clearly the **overall batch of educational content** progressed. The **Pacing & Flow** metric reflects the logical progression and rhythm of lessons across the set.

    **Evaluation Methodology:**
    - **Average Similarity (−1 to 1):** Measures how well consecutive segments connect; higher values reflect smoother transitions.
    - **Similarity Standard Deviation (≥0):** Captures flow stability; lower values indicate consistent pacing.
    - **Consistency Score (0 to 1):** Derived from deviation, indicating steadiness of structure.
    - **Raw Flow Score (−10 to 10):** Aggregates the above into a directional pacing score.
    - **Final Ratings (1–5):** Higher values reflect smoother, more cohesive delivery.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence holistic summary, mostly in passive voice.
    - Focus: Overall sense of pacing—whether the flow felt natural, abrupt, repetitive, or hard to follow.

    **Prompt Instruction:**
    Based on the following batch-level values:
    - Avg Similarity values: {avg_similarity_list}
    - Similarity Std Dev values: {similarity_std_list}
    - Consistency Scores: {consistency_score_list}
    - Raw Flow Scores: {raw_score_list}
    - Final Ratings: {final_rating_list}

    Write a learner-style reflection describing how the **content flowed overall**. Avoid metric names or numbers in the answer. Describe whether the lessons generally felt smooth and easy to follow, or if they felt disjointed or inconsistent.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **pacing and instructional flow** across a **batch of educational content** using the **Pacing & Flow Score**.

    **Evaluation Methodology:**
    - **Average Similarity (−1 to 1):** Measures how well consecutive segments connect; higher values reflect smoother transitions.
    - **Similarity Standard Deviation (≥0):** Captures flow stability; lower values indicate consistent pacing.
    - **Consistency Score (0 to 1):** Derived from deviation, indicating steadiness of structure.
    - **Raw Flow Score (−10 to 10):** Aggregates the above into a directional pacing score.
    - **Final Ratings (1–5):** Higher values reflect smoother, more cohesive delivery.

    **Evaluation Metrics Overview:**
    - Average Similarity (−1 to 1): {avg_similarity_list}
    - Similarity Std Deviation (≥0): {similarity_std_list}
    - Consistency Score (0 to 1): {consistency_score_list}
    - Raw Flow Score (−10 to 10): {raw_score_list}
    - Final Ratings (1–5): {final_rating_list}

    **Instruction:**
    Based on these batch-level metrics, write a 2–4 line instructor-style summary.  
    Identify which metric showed the greatest deviation from ideal targets and recommend improvements accordingly.  
    The tone should be formal and passive, with a focus on **general structural weaknesses** such as lack of continuity, abrupt transitions, or inconsistent pacing.  
    Avoid per-sample feedback; provide batch-level insight only.
    """
    return learner_prompt, instructor_prompt

def prompt_pacing_flow_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how smoothly and clearly the **course** progressed across its modules. The **Pacing & Flow** metric reflects the logical progression and rhythm of lessons from start to finish.

    **Evaluation Methodology:**
    - **Average Similarity (−1 to 1):** Higher values suggest smoother transitions between segments.
    - **Similarity Standard Deviation (≥0):** Lower values suggest stable pacing.
    - **Consistency Score (0 to 1):** Higher values indicate steady structure.
    - **Raw Flow Score (−10 to 10):** Overall directional pacing measure.
    - **Final Ratings (1–5):** Higher means smoother, more cohesive delivery.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence holistic summary using passive voice where natural.
    - Focus: Whether the pacing felt smooth, abrupt, repetitive, or inconsistent across the course.

    **Prompt Instruction:**
    Given the following **module-level course data**:

    {module_level_data}

    Write a short learner-style reflection describing the **overall flow of the course**. Avoid technical terms or metric names. Focus on the general sense of ease or difficulty in following the lessons from start to finish.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **pacing and instructional flow** for a course composed of multiple modules. The **Pacing & Flow** metric measures the smoothness and consistency of lesson delivery across the course.

    **Evaluation Methodology:**
    - **Average Similarity (−1 to 1):** Smoother transitions at higher values.
    - **Similarity Standard Deviation (≥0):** Lower indicates steadier pacing.
    - **Consistency Score (0 to 1):** Higher reflects more stable structure.
    - **Raw Flow Score (−10 to 10):** Aggregated pacing performance.
    - **Final Ratings (1–5):** Higher reflects smoother delivery.

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Provide a 2–4 line course-level summary.
    - Identify the most significant pacing or flow weakness observed across modules.
    - Offer one actionable recommendation to improve pacing and transitions.
    - Avoid module-specific commentary; focus on overarching trends.

    Write a concise, high-level evaluation of the course’s pacing and flow quality.
    """
    
    return learner_prompt, instructor_prompt

def prompt_well_structured(unique_clusters, total_clusters, cluster_diversity, raw_score, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well the course content is organized. The **Well Structured** metric reflects how clearly themes and topics are grouped, helping learners navigate and retain the material effectively.

    **Evaluation Methodology:**
    - **Unique Clusters Found:** Number of distinct content themes detected.
    - **Total Possible Clusters:** Max number of meaningful clusters the course could achieve.
    - **Cluster Diversity (0–1):** Ratio of actual vs. possible clusters — higher means better structure.
    - **Raw Structure Score (0–10):** Aggregated signal of thematic clarity and segmentation.
    - **Final Rating:** Scaled from 1 (poorly structured) to 5 (highly structured and organized).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: How structured and navigable the course felt during learning.

    **Prompt Instruction:**
    Given Unique Clusters: {unique_clusters}, Total Clusters: {total_clusters}, Cluster Diversity: {cluster_diversity}, Raw Score: {raw_score}/10, and Final Rating: {final_rating}, write a short learner-style reflection on the perceived structure and organization of the course. Highlight how easy or difficult it was to follow the flow of topics.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the structural clarity of a course using the **Well Structured Score**, derived from clustering-based analysis.

    **Methodology & Overview:**
    - Unique Clusters Found (Range: 1–{total_clusters}): Indicates distinct content areas.
    - Cluster Diversity (0 to 1): Reflects proportional thematic spread.
    - Raw Structure Score (0–10): Reflects separation strength among clusters.
    - Final Rating (1–5): Computed algorithmically from the above.

    **Evaluation Input:**
    Unique Clusters: {unique_clusters},
    Total Clusters: {total_clusters},
    Cluster Diversity: {cluster_diversity},
    Raw Score: {raw_score}/10,
    Final Rating: {final_rating}.

    **Response Format (2–4 lines, passive voice):**  
    Identify which metric deviates most from the ideal. Offer concise structural improvement suggestions in passive tone.  
    Keep it formal, focused, and free of filler — suitable for direct communication with the instructor.
    """
    return learner_prompt, instructor_prompt

def prompt_well_structured_batch(unique_clusters_list, total_clusters_list, cluster_diversity_list, raw_score_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well-organized the course content felt across a batch. The **Well Structured** metric reflects how clearly topics were grouped and how intuitively the material flowed.

    **Evaluation Methodology:**
    - **Unique Clusters Found:** Number of distinct content themes detected.
    - **Total Possible Clusters:** Theoretical max clusters.
    - **Cluster Diversity (0–1):** Measures content variety and thematic balance.
    - **Raw Structure Score (0–10):** Indicates how clearly themes were segmented.
    - **Final Rating:** Scaled from 1 (poor structure) to 5 (strong structure).

    **Expected Response Guidelines:**
    - Perspective: Simulated learner.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence batch-level summary.
    - Focus: Overall navigability and flow across all evaluated content.

    **Prompt Instruction:**
    You are given data from multiple learning modules:
    - Unique Clusters Found: {unique_clusters_list}
    - Total Possible Clusters: {total_clusters_list}
    - Cluster Diversity: {cluster_diversity_list}
    - Raw Structure Scores: {raw_score_list}
    - Final Ratings (1–5): {final_rating_list}

    Write a short reflection from the learner’s point of view that summarizes the **overall perception**. Avoid technical breakdowns — instead, express how well the learning experience flowed and whether the themes felt clear and well-paced across the set.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **structural clarity** of a batch using clustering-based analytics.

    **Evaluation Methodology:**
    - **Unique Clusters Found:** Number of distinct content themes detected.
    - **Total Possible Clusters:** Theoretical max clusters.
    - **Cluster Diversity (0–1):** Measures content variety and thematic balance.
    - **Raw Structure Score (0–10):** Indicates how clearly themes were segmented.
    - **Final Rating:** Scaled from 1 (poor structure) to 5 (strong structure).

    **Input Metrics Across the Batch:**
    - Unique Clusters Found: {unique_clusters_list}
    - Total Possible Clusters: {total_clusters_list}
    - Cluster Diversity (0–1): {cluster_diversity_list}
    - Raw Structure Score (0–10): {raw_score_list}
    - Final Rating (1–5): {final_rating_list}

    **Instructions for Feedback:**
    Review the metrics and provide a 2–4 line holistic summary.  
    Focus on the metric that most consistently deviates from its ideal value (e.g., low diversity or weak raw structure).  
    Provide improvement-focused suggestions in passive voice, suitable for instructors reviewing content structure.  
    Avoid course-specific details — stick to batch-level trends and recommendations.
    """
    return learner_prompt, instructor_prompt

def prompt_well_structured_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well-organized the course content felt across its modules. The **Well Structured** metric reflects how clearly topics were grouped and how intuitively the material flowed throughout the course.

    **Evaluation Methodology:**
    - **Unique Clusters Found:** Number of distinct content themes detected.
    - **Total Possible Clusters:** Theoretical maximum themes.
    - **Cluster Diversity (0–1):** Measures content variety and thematic balance.
    - **Raw Structure Score (0–10):** Indicates clarity of theme segmentation.
    - **Final Rating:** Scaled from 1 (poor structure) to 5 (strong structure).

    **Expected Response Guidelines:**
    - Perspective: Simulated learner.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence reflection using passive voice where natural.
    - Focus: Overall navigability, thematic clarity, and pacing across the course.

    **Prompt Instruction:**
    Given the following **module-level course data**, reflect on the **overall course-level** experience:

    {module_level_data}

    Write a short learner-style reflection that expresses how well the learning journey flowed and whether the themes felt clear and well-paced. Avoid technical or metric-heavy commentary.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **structural clarity** of a course made up of several modules, using clustering-based analytics.

    **Evaluation Methodology:**
    - **Unique Clusters Found:** Number of distinct content themes detected.
    - **Total Possible Clusters:** Theoretical maximum.
    - **Cluster Diversity (0–1):** Measures variety and thematic balance.
    - **Raw Structure Score (0–10):** Reflects clarity of theme segmentation.
    - **Final Rating:** Scaled from 1 (poor structure) to 5 (strong structure).

    **Course Module Data:**
    {module_level_data}

    **Feedback Instructions:**
    - Write a 2–4 line summary in formal tone and passive voice.
    - Identify the structural metric most consistently deviating from its ideal.
    - Suggest a single, course-level improvement direction without going into per-module detail.
    - Keep recommendations focused on overall thematic clarity, navigability, and flow.

    Provide a concise, improvement-focused summary for instructors to enhance structural organization in future iterations.
    """

    return learner_prompt, instructor_prompt

def prompt_learning_value(avg_coherence, coherence_std, focus_consistency, raw_score, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well the course stayed focused and delivered on its educational promise. The **Learning Value** score reflects the thematic coherence and consistency of the content across the course.

    **Evaluation Methodology:**
    - **Average Coherence (-1 to 1):** Indicates how well segments relate to the main theme.
    - **Coherence Standard Deviation (≥0):** Lower values suggest consistent focus.
    - **Focus Consistency (0–1):** Higher values mean fewer digressions and more clarity.
    - **Raw Learning Score (-10 to 10):** Synthesizes all aspects of structural alignment.
    - **Final Rating:** A 1–5 score where 5 reflects highly focused, cohesive learning.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: How coherent and focused the content felt while progressing through the content.

    **Prompt Instruction:**
    Given Avg Coherence: {avg_coherence}, Coherence Std Dev: {coherence_std}, Focus Consistency: {focus_consistency}, Raw Score: {raw_score}/10, and Final Score: {final_rating}, write a short learner-style reflection on how focused and educationally valuable the course felt. Reflect whether the course stayed on topic and maintained a clear learning direction.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Learning Value Score** of educational content. This score reflects how well the content maintains focus and delivers educational value.

    **Methodology & Overview:**
    - **Average Coherence** (Range: -1 to 1): Higher indicates focused delivery.
    - **Coherence Standard Deviation** (≥0): Lower suggests uniform thematic focus.
    - **Focus Consistency** (Range: 0–1): Measures content drift; 1 is ideal.
    - **Raw Learning Score** (Range: -10 to 10): Captures comprehensive educational value.

    **Final Score Rationale:**
    The final score (1 to 5) is calculated by aggregating all intermediate metrics without arbitrary thresholds. A score of 5 represents consistently high-value, focused instruction; a score of 1 indicates major thematic drift or fragmentation.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Average Coherence: {avg_coherence}, 
    Coherence Std Dev: {coherence_std},  
    Focus Consistency: {focus_consistency}, 
    Raw Learning Score: {raw_score}/10, 
    Final Rating: {final_rating}.

    Please provide a concise 2–4 line summary, identifying which metric deviates most from ideal values.  
    Suggestions should be framed in passive voice and focused on improving coherence, consistency, or depth to raise learning effectiveness.
    """
    return learner_prompt, instructor_prompt

def prompt_learning_value_batch(avg_coherence_list, coherence_std_list, focus_consistency_list, raw_score_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well a batch maintained focus and delivered meaningful educational value. The **Learning Value Score** reflects coherence, consistency, and clarity across multiple learning journeys.

    **Evaluation Methodology:**
    - **Average Coherence (-1 to 1):** Higher values indicate strong thematic alignment.
    - **Coherence Standard Deviation (≥0):** Lower values imply consistent delivery and fewer off-topic detours.
    - **Focus Consistency (0–1):** Measures drift and cohesion across sections—closer to 1 indicates more clarity.
    - **Raw Learning Score (-10 to 10):** Synthesizes alignment and value across learning components.
    - **Final Rating (1–5):** Reflects holistic educational effectiveness and content focus.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner's voice.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary of batch-wide learning experience.
    - Focus: How the **overall set** felt in terms of focus, clarity, and educational value.

    **Prompt Instruction:**
    Given the following batch-level metrics:
    - Average Coherence values: {avg_coherence_list}
    - Coherence Standard Deviations: {coherence_std_list}
    - Focus Consistency values: {focus_consistency_list}
    - Raw Learning Scores: {raw_score_list}
    - Final Ratings (1–5): {final_rating_list}

    Write a short learner-style reflection summarizing the overall educational value and thematic clarity across the batch. Avoid numeric or technical explanations—focus on how focused and coherent the learning experiences *felt* across the board.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Learning Value Score** across a batch of educational contents. This score quantifies the degree of focus, clarity, and thematic alignment.

    **Metric Overview:**
    - **Average Coherence** (-1 to 1): Measures alignment to central topic.
    - **Coherence Std Dev** (≥0): Highlights fluctuation in topic focus.
    - **Focus Consistency** (0–1): Tracks digression and clarity.
    - **Raw Learning Score** (-10 to 10): Represents integrated value.
    - **Final Rating (1–5):** Aggregated outcome reflecting  educational effectiveness.

    **Batch-Level Values:**
    - Avg Coherence: {avg_coherence_list}
    - Coherence Std Dev: {coherence_std_list}
    - Focus Consistency: {focus_consistency_list}
    - Raw Scores: {raw_score_list}
    - Final Ratings: {final_rating_list}

    **Instruction:**
    Provide a 2–4 line summary assessing overall learning value across the set.  
    Highlight which metric shows the greatest deviation from its ideal value (Coherence → 1, Std Dev → 0, Focus → 1, Raw Score → 10).  
    Frame the summary in passive voice, and offer concise improvement-focused feedback related to content clarity, consistency, or thematic discipline.
    """
    return learner_prompt, instructor_prompt

def prompt_learning_value_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well a **course** maintained focus and delivered meaningful educational value across its modules. The **Learning Value Score** reflects coherence, consistency, and clarity throughout the learning journey.

    **Evaluation Methodology:**
    - **Average Coherence (-1 to 1):** Higher values indicate strong thematic alignment.
    - **Coherence Standard Deviation (≥0):** Lower values imply consistent delivery and fewer off-topic detours.
    - **Focus Consistency (0–1):** Measures drift and cohesion—closer to 1 indicates more clarity.
    - **Raw Learning Score (-10 to 10):** Synthesizes alignment and value across learning components.
    - **Final Learning Value Rating (1–5):** Reflects holistic educational effectiveness and content focus.

    **Expected Response Guidelines:**
    - Perspective: Learner’s point of view (not technical/AI).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence reflection using passive voice where natural.
    - Focus: Overall clarity, focus, and educational value across the course.

    **Prompt Instruction:**
    Given the following **module-level data**, write a learner-style reflection on the **overall course-level experience** in terms of thematic clarity, focus, and educational effectiveness:

    {module_level_data}

    Avoid numeric or technical detail—focus on how the course felt in terms of focus and coherence to the learner.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Learning Value Score** across a course composed of multiple modules. This score quantifies the degree of focus, clarity, and thematic alignment at the course level.

    **Metric Overview:**
    - **Average Coherence (-1 to 1):** Ideal = 1
    - **Coherence Std Dev (≥0):** Ideal = 0
    - **Focus Consistency (0–1):** Ideal = 1
    - **Raw Learning Score (-10 to 10):** Ideal = 10
    - **Final Learning Value Rating (1–5):** Reflects overall educational effectiveness

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a 2–4 line instructor-style summary assessing overall learning value.
    - Identify which metric shows the greatest deviation from its ideal value.
    - Offer concise, actionable feedback related to improving clarity, consistency, or thematic focus.
    - Avoid module-specific breakdowns; focus on course-wide trends.

    Provide a high-level evaluation of the course’s learning value.
    """
    
    return learner_prompt, instructor_prompt

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

def evaluate_transcript_content(file_path: str, service_account_file: str, file_name: str) -> dict:
    print(f"File Name: {file_name}")
    text = read_transcript(file_path, service_account_file)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 4]
    embeddings = t_model.encode(sentences, convert_to_tensor=True, batch_size=64, device="cuda")

    # Calculate all metrics with intermediate outputs
    easy_result = calculate_easy_to_understand_score(doc)
    engagement_result = classify_engagement_feel_dynamic(sentences)
    pacing_result = pacing_and_flow_score(sentences, embeddings)
    structure_result = is_course_well_structured_score(sentences, embeddings)
    learning_result = learning_value_score(sentences, embeddings)

    # Generate user and instructor feedback using prompt functions
    easy_score_user_feedback = execute_prompt(prompt_easy_to_understand(
        easy_result['intermediates']['word_length_avg'],
        easy_result['intermediates']['syllables_avg'],
        easy_result['intermediates']['raw_score'],
        easy_result['score']
    )[0])

    easy_score_instructor_feedback = execute_prompt(prompt_easy_to_understand(
        easy_result['intermediates']['word_length_avg'],
        easy_result['intermediates']['syllables_avg'],
        easy_result['intermediates']['raw_score'],
        easy_result['score']
    )[1])

    engagement_score_user_feedback = execute_prompt(prompt_engagement(
        engagement_result['intermediates']['avg_sentiment'],
        engagement_result['intermediates']['positive_ratio'],
        engagement_result['intermediates']['raw_score'],
        engagement_result['score']
    )[0])

    engagement_score_instructor_feedback = execute_prompt(prompt_engagement(
        engagement_result['intermediates']['avg_sentiment'],
        engagement_result['intermediates']['positive_ratio'],
        engagement_result['intermediates']['raw_score'],
        engagement_result['score']
    )[1])

    pacing_score_user_feedback = execute_prompt(prompt_pacing_flow(
        pacing_result['intermediates']['avg_similarity'],
        pacing_result['intermediates']['similarity_std'],
        pacing_result['intermediates']['consistency_score'],
        pacing_result['intermediates']['raw_score'],
        pacing_result['score']
    )[0])

    pacing_score_instructor_feedback = execute_prompt(prompt_pacing_flow(
        pacing_result['intermediates']['avg_similarity'],
        pacing_result['intermediates']['similarity_std'],
        pacing_result['intermediates']['consistency_score'],
        pacing_result['intermediates']['raw_score'],
        pacing_result['score']
    )[1])

    structure_score_user_feedback = execute_prompt(prompt_well_structured(
        structure_result['intermediates']['unique_clusters'],
        structure_result['intermediates']['total_clusters'],
        structure_result['intermediates']['cluster_diversity'],
        structure_result['intermediates']['raw_score'],
        structure_result['score']
    )[0])

    structure_score_instructor_feedback = execute_prompt(prompt_well_structured(
        structure_result['intermediates']['unique_clusters'],
        structure_result['intermediates']['total_clusters'],
        structure_result['intermediates']['cluster_diversity'],
        structure_result['intermediates']['raw_score'],
        structure_result['score']
    )[1])

    learning_score_user_feedback = execute_prompt(prompt_learning_value(
        learning_result['intermediates']['avg_coherence'],
        learning_result['intermediates']['coherence_std'],
        learning_result['intermediates']['focus_consistency'],
        learning_result['intermediates']['raw_score'],
        learning_result['score']
    )[0])

    learning_score_instructor_feedback = execute_prompt(prompt_learning_value(
        learning_result['intermediates']['avg_coherence'],
        learning_result['intermediates']['coherence_std'],
        learning_result['intermediates']['focus_consistency'],
        learning_result['intermediates']['raw_score'],
        learning_result['score']
    )[1])

    return {
        "File Name": file_name,
        "Easy-to-Understand Score": {
            "Intermediate Parameters": {
                "Average Word Length": easy_result['intermediates']['word_length_avg'],
                "Average Syllables per word": easy_result['intermediates']['syllables_avg'],
                "Raw Score": easy_result['intermediates']['raw_score']
            },
            "Final Score": easy_result['score'],
            "User Persepective Assessment": easy_score_user_feedback,
            "Instructor Feedback": easy_score_instructor_feedback
        },
        "Engagement Score": {
            "Intermediate Parameters": {
                "Average Sentiment": engagement_result['intermediates']['avg_sentiment'],
                "Positive Ratio": engagement_result['intermediates']['positive_ratio'],
                "Raw Score": engagement_result['intermediates']['raw_score']
            },
            "Final Score": engagement_result['score'],
            "User Persepective Assessment": engagement_score_user_feedback,
            "Instructor Feedback": engagement_score_instructor_feedback
        },
        "Pacing & Flow Score": {
            "Intermediate Parameters": {
                "Average Similarity": pacing_result['intermediates']['avg_similarity'],
                "Similarity Std Dev": pacing_result['intermediates']['similarity_std'],
                "Consistency Score": pacing_result['intermediates']['consistency_score'],
                "Raw Score": pacing_result['intermediates']['raw_score']
            },
            "Final Score": pacing_result['score'],
            "User Persepective Assessment": pacing_score_user_feedback,
            "Instructor Feedback": pacing_score_instructor_feedback
        },
        "Well Structured Score": {
            "Intermediate Parameters": {
                "Unique Clusters": structure_result['intermediates']['unique_clusters'],
                "Total Clusters": structure_result['intermediates']['total_clusters'],
                "Cluster Diversity": structure_result['intermediates']['cluster_diversity'],
                "Raw Score": structure_result['intermediates']['raw_score']
            },
            "Final Score": structure_result['score'],
            "User Persepective Assessment": structure_score_user_feedback,
            "Instructor Feedback": structure_score_instructor_feedback
        },
        "Learning Value Score": {
            "Intermediate Parameters": {
                "Average Coherence": learning_result['intermediates']['avg_coherence'],
                "Coherence Std Dev": learning_result['intermediates']['coherence_std'],
                "Focus Consistency": learning_result['intermediates']['focus_consistency'],
                "Raw Score": learning_result['intermediates']['raw_score']
            },
            "Final Score": learning_result['score'],
            "User Persepective Assessment": learning_score_user_feedback,
            "Instructor Feedback": learning_score_instructor_feedback
        }
    }

def evaluate_reading_content(file_name, text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 4]
    embeddings = t_model.encode(sentences, convert_to_tensor=True, batch_size=64, device="cuda")

    # Calculate all metrics with intermediate outputs
    easy_result = calculate_easy_to_understand_score(doc)
    engagement_result = classify_engagement_feel_dynamic(sentences)
    pacing_result = pacing_and_flow_score(sentences, embeddings)
    structure_result = is_course_well_structured_score(sentences, embeddings)
    learning_result = learning_value_score(sentences, embeddings)

    # Generate user and instructor feedback using prompt functions
    easy_score_user_feedback = execute_prompt(prompt_easy_to_understand(
        easy_result['intermediates']['word_length_avg'],
        easy_result['intermediates']['syllables_avg'],
        easy_result['intermediates']['raw_score'],
        easy_result['score']
    )[0])

    easy_score_instructor_feedback = execute_prompt(prompt_easy_to_understand(
        easy_result['intermediates']['word_length_avg'],
        easy_result['intermediates']['syllables_avg'],
        easy_result['intermediates']['raw_score'],
        easy_result['score']
    )[1])

    engagement_score_user_feedback = execute_prompt(prompt_engagement(
        engagement_result['intermediates']['avg_sentiment'],
        engagement_result['intermediates']['positive_ratio'],
        engagement_result['intermediates']['raw_score'],
        engagement_result['score']
    )[0])

    engagement_score_instructor_feedback = execute_prompt(prompt_engagement(
        engagement_result['intermediates']['avg_sentiment'],
        engagement_result['intermediates']['positive_ratio'],
        engagement_result['intermediates']['raw_score'],
        engagement_result['score']
    )[1])

    pacing_score_user_feedback = execute_prompt(prompt_pacing_flow(
        pacing_result['intermediates']['avg_similarity'],
        pacing_result['intermediates']['similarity_std'],
        pacing_result['intermediates']['consistency_score'],
        pacing_result['intermediates']['raw_score'],
        pacing_result['score']
    )[0])

    pacing_score_instructor_feedback = execute_prompt(prompt_pacing_flow(
        pacing_result['intermediates']['avg_similarity'],
        pacing_result['intermediates']['similarity_std'],
        pacing_result['intermediates']['consistency_score'],
        pacing_result['intermediates']['raw_score'],
        pacing_result['score']
    )[1])

    structure_score_user_feedback = execute_prompt(prompt_well_structured(
        structure_result['intermediates']['unique_clusters'],
        structure_result['intermediates']['total_clusters'],
        structure_result['intermediates']['cluster_diversity'],
        structure_result['intermediates']['raw_score'],
        structure_result['score']
    )[0])

    structure_score_instructor_feedback = execute_prompt(prompt_well_structured(
        structure_result['intermediates']['unique_clusters'],
        structure_result['intermediates']['total_clusters'],
        structure_result['intermediates']['cluster_diversity'],
        structure_result['intermediates']['raw_score'],
        structure_result['score']
    )[1])

    learning_score_user_feedback = execute_prompt(prompt_learning_value(
        learning_result['intermediates']['avg_coherence'],
        learning_result['intermediates']['coherence_std'],
        learning_result['intermediates']['focus_consistency'],
        learning_result['intermediates']['raw_score'],
        learning_result['score']
    )[0])

    learning_score_instructor_feedback = execute_prompt(prompt_learning_value(
        learning_result['intermediates']['avg_coherence'],
        learning_result['intermediates']['coherence_std'],
        learning_result['intermediates']['focus_consistency'],
        learning_result['intermediates']['raw_score'],
        learning_result['score']
    )[1])

    return {
        "File Name": file_name,
        "Easy-to-Understand Score": {
            "Intermediate Parameters": {
                "Average Word Length": easy_result['intermediates']['word_length_avg'],
                "Average Syllables per word": easy_result['intermediates']['syllables_avg'],
                "Raw Score": easy_result['intermediates']['raw_score']
            },
            "Final Score": easy_result['score'],
            "User Persepective Assessment": easy_score_user_feedback,
            "Instructor Feedback": easy_score_instructor_feedback
        },
        "Engagement Score": {
            "Intermediate Parameters": {
                "Average Sentiment": engagement_result['intermediates']['avg_sentiment'],
                "Positive Ratio": engagement_result['intermediates']['positive_ratio'],
                "Raw Score": engagement_result['intermediates']['raw_score']
            },
            "Final Score": engagement_result['score'],
            "User Persepective Assessment": engagement_score_user_feedback,
            "Instructor Feedback": engagement_score_instructor_feedback
        },
        "Pacing & Flow Score": {
            "Intermediate Parameters": {
                "Average Similarity": pacing_result['intermediates']['avg_similarity'],
                "Similarity Std Dev": pacing_result['intermediates']['similarity_std'],
                "Consistency Score": pacing_result['intermediates']['consistency_score'],
                "Raw Score": pacing_result['intermediates']['raw_score']
            },
            "Final Score": pacing_result['score'],
            "User Persepective Assessment": pacing_score_user_feedback,
            "Instructor Feedback": pacing_score_instructor_feedback
        },
        "Well Structured Score": {
            "Intermediate Parameters": {
                "Unique Clusters": structure_result['intermediates']['unique_clusters'],
                "Total Clusters": structure_result['intermediates']['total_clusters'],
                "Cluster Diversity": structure_result['intermediates']['cluster_diversity'],
                "Raw Score": structure_result['intermediates']['raw_score']
            },
            "Final Score": structure_result['score'],
            "User Persepective Assessment": structure_score_user_feedback,
            "Instructor Feedback": structure_score_instructor_feedback
        },
        "Learning Value Score": {
            "Intermediate Parameters": {
                "Average Coherence": learning_result['intermediates']['avg_coherence'],
                "Coherence Std Dev": learning_result['intermediates']['coherence_std'],
                "Focus Consistency": learning_result['intermediates']['focus_consistency'],
                "Raw Score": learning_result['intermediates']['raw_score']
            },
            "Final Score": learning_result['score'],
            "User Persepective Assessment": learning_score_user_feedback,
            "Instructor Feedback": learning_score_instructor_feedback
        }
    }

def chunk_transcript(text, chunk_size=250, overlap=50):
    """Split transcript into overlapping chunks for better analysis"""
    if not text:
        return []

    # First try sentence-based chunking
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) <= chunk_size:
            current_chunk.append(sentence)
            current_length += len(words)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # If we couldn't create meaningful chunks, fall back to word-based chunking
    if not chunks:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i + chunk_size]
            if len(chunk) > 10:  # Only include chunks with meaningful content
                chunks.append(' '.join(chunk))

    return chunks

def calculate_semantic_similarity(los, transcript_chunks, t_model):
    """Calculate semantic similarity between LOs and transcript chunks using GPU acceleration"""
    if not los or not transcript_chunks:
        return np.zeros((len(los) if los else 0, 1))

    # Encode LOs and transcript chunks with batch processing for GPU efficiency
    lo_embeddings = t_model.encode(los, convert_to_tensor=True, batch_size=64, device="cuda")
    chunk_embeddings = t_model.encode(transcript_chunks, convert_to_tensor=True, batch_size=64, device="cuda")

    # Calculate cosine similarity using efficient tensor operations
    if torch.cuda.is_available():
        # Keep computation on GPU for speed
        similarity_matrix = torch.mm(lo_embeddings, chunk_embeddings.transpose(0, 1))
        return similarity_matrix.cpu().numpy()
    else:
        # Fall back to CPU if needed
        return torch.mm(lo_embeddings, chunk_embeddings.transpose(0, 1)).numpy()

def calculate_keyword_overlap(los, transcript_text):
    """Calculate keyword overlap between LOs and transcript with intermediate results"""
    if not los or not transcript_text:
        return {
            'score': 1,
            'intermediates': {
                'overlap_score': 0.0,
                'keyword_hits': 0,
                'total_keywords': 0,
                'top_keywords': []
            }
        }

    # Extract keywords using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

    # Combine LOs into a single document for vectorization
    combined_los = " ".join(los)

    # Fit and transform
    try:
        tfidf_matrix = vectorizer.fit_transform([combined_los, transcript_text])

        # Get feature names (keywords)
        feature_names = vectorizer.get_feature_names_out()

        # Get top keywords from LOs - optimize array operations
        lo_tfidf = tfidf_matrix[0].toarray()[0]

        # Use numpy for faster sorting and filtering
        top_indices = np.argsort(lo_tfidf)[-20:]  # Top 20 keywords
        lo_keywords = {feature_names[i] for i in top_indices if lo_tfidf[i] > 0}
        top_keywords_list = [feature_names[i] for i in top_indices if lo_tfidf[i] > 0]

        # Count keyword occurrences in transcript - use set operations for speed
        transcript_text_lower = transcript_text.lower()
        keyword_hits = sum(1 for word in lo_keywords if word.lower() in transcript_text_lower)

        # Calculate overlap score (0-1 range)
        overlap_score = keyword_hits / len(lo_keywords) if lo_keywords else 0

        # Convert to integer 1-5 scale
        score = round(float(np.clip(overlap_score * 4 + 1, 1, 5)), 2)

        return {
            'score': score,
            'intermediates': {
                'overlap_score': round(overlap_score, 3),
                'keyword_hits': keyword_hits,
                'total_keywords': len(lo_keywords),
                'top_keywords': top_keywords_list
            }
        }
    except Exception as e:
        print(f"Error in keyword overlap calculation: {e}")
        return {
            'score': 1,
            'intermediates': {
                'overlap_score': 0.0,
                'keyword_hits': 0,
                'total_keywords': 0,
                'top_keywords': []
            }
        }

def prompt_semantic_alignment(semantic_alignment_raw, max_similarities, mean_similarities, num_chunks, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well the course content reflects its stated learning objectives. The **Semantic Alignment** metric captures the meaningful overlap between objectives and actual transcript content.

    **Evaluation Methodology:**
    - **Semantic Alignment Raw Score (0–1):** Average of maximum semantic similarities between learning objectives and transcript segments.
    - **Maximum vs Mean Similarities:** Higher values (closer to 1) suggest better coverage and reinforcement of objectives.
    - **Content Chunks Analyzed:** Indicates granularity of coverage detection.
    - **Final Rating (1–5):** Represents depth of alignment from poor (1) to excellent (5).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Whether objectives felt clearly and consistently covered across the content.

    **Prompt Instruction:**
    Given Raw Score: {semantic_alignment_raw}, Maximum Similarities: {max_similarities}, Mean Similarities: {mean_similarities}, Content Chunks: {num_chunks}, and Final Score: {final_rating}, write a short learner-style reflection on how well the content addressed the learning objectives. Reference the intermediate metrics and clarify perceived strength or weakness in alignment.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Semantic Alignment** between curriculum learning objectives and instructional content.

    **Methodology & Overview:**
    - **Semantic Alignment Raw Score** (0–1): Aggregates the strongest cosine similarities between objectives and transcript chunks.
    - **Maximum Similarities** reveal high-coverage objectives; **Mean Similarities** reflect how consistently each objective is addressed.
    - **Content Chunk Count** indicates analytical resolution, ensuring no objective is overlooked.

    **Final Score Calculation:**
    The aggregated alignment score is linearly mapped to a 1–5 scale. A 5 reflects thorough and even alignment, while a 1 represents major curriculum-content disconnect.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Semantic Alignment Raw Score: {semantic_alignment_raw}, 
    Max Similarities: {max_similarities}, 
    Mean Similarities: {mean_similarities}, 
    Content Chunks: {num_chunks}, 
    Final Rating: {final_rating}.

    Please provide a 2–4 line summary in passive voice identifying the weakest alignment signal. Suggestions should focus on improving coverage or consistency for under-aligned learning objectives.
    """

    return learner_prompt, instructor_prompt

def prompt_semantic_alignment_batch(semantic_alignment_raw_list, max_similarities_list, mean_similarities_list, num_chunks_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well a **batch of learning files** reflected their stated learning objectives. The **Semantic Alignment** metric reflects overlapping meaning between objectives and transcripted content across the set.

    **Evaluation Methodology (Across the Batch):**
    - **Semantic Alignment Raw Score (0–1):** Average of maximum cosine similarities per file.
    - **Maximum vs Mean Similarities:** Higher, more consistent values indicate objectives are well covered across content.
    - **Content Chunks Analyzed:** Indicates coverage granularity.
    - **Final Ratings (1–5):** Scaled alignment scores for each file.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence holistic summary.
    - Focus: Whether objectives *felt* clearly and consistently covered across the batch.

    **Prompt Instruction:**
    Given the following aggregate inputs:
    - Raw Scores: {semantic_alignment_raw_list}
    - Max Similarities: {max_similarities_list}
    - Mean Similarities: {mean_similarities_list}
    - Chunk Counts: {num_chunks_list}
    - Final Ratings: {final_rating_list}

    Write a short learner-style reflection describing the *overall impression* of how well the learning objectives were reflected across the batch. Avoid technical wording and focus on how strongly and consistently the objectives appeared to guide the content.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Semantic Alignment** between learning objectives and transcript content across a batch of files.

    **Evaluation Methodology (Across the Batch):**
    - **Semantic Alignment Raw Score (0–1):** Average of maximum cosine similarities per file.
    - **Maximum vs Mean Similarities:** Higher, more consistent values indicate objectives are well covered across content.
    - **Content Chunks Analyzed:** Indicates coverage granularity.
    - **Final Ratings (1–5):** Scaled alignment scores for each file.

    **Batch Metrics:**
    - Raw Alignment Scores: {semantic_alignment_raw_list}
    - Max Similarities: {max_similarities_list}
    - Mean Similarities: {mean_similarities_list}
    - Chunk Counts: {num_chunks_list}
    - Final Ratings (1–5): {final_rating_list}

    Provide a concise (2–4 lines), passive-voice summary of the **overall alignment quality** across the batch.  
    Identify which metric most consistently deviated from ideal values (Raw, Max, Mean, or Chunk Count).  
    Suggestions should focus on improving coverage or reinforcement of under-aligned learning objectives without sample-level commentary.
    """

    return learner_prompt, instructor_prompt

def prompt_semantic_alignment_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well a **course** reflected its stated learning objectives across multiple modules.  
    The **Semantic Alignment** metric reflects how closely the meaning of the transcript content aligns with the intended objectives.

    **Evaluation Methodology:**
    - **Max Similarities**: Higher values indicate strong alignment between objectives and the most relevant parts of the content.
    - **Mean Similarities**: Higher averages suggest consistent reinforcement of objectives.
    - **Number of Chunks**: Indicates granularity of coverage within the content.
    - **Raw Scores**: Overall semantic alignment strength on a 0–1 scale.
    - **Final Ratings (1–5)**: Learner-friendly alignment scores per module.

    **Expected Response Guidelines:**
    - Perspective: From a learner’s point of view (not technical).
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence course-level reflection.
    - Focus: Overall clarity, consistency, and relevance of content to stated objectives.

    **Prompt Instruction:**
    Given the following **module-level data**:

    {module_level_data}

    Write a short learner-style reflection on the **overall course-level impression** of how well the learning objectives were reflected.  
    Avoid technical terms and module-by-module commentary — focus on the general sense of objective coverage and consistency.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **Semantic Alignment** between learning objectives and transcript content for a course comprising multiple modules.

    **Evaluation Methodology:**
    - **Max Similarities**: Measures how closely any part of the content aligns with each objective.
    - **Mean Similarities**: Indicates average reinforcement of objectives across content.
    - **Number of Chunks**: Reflects the granularity of analysis.
    - **Raw Scores**: Strength of semantic match on a 0–1 scale.
    - **Final Ratings (1–5)**: Scaled alignment scores per module.

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a 2–4 line course-level summary of **overall alignment quality**.
    - Identify which metric most consistently deviated from ideal expectations.
    - Provide a concise, actionable recommendation for improving alignment coverage or reinforcement of objectives.
    - Avoid module-specific detail — focus on overall trends.

    Provide a high-level evaluation of semantic alignment across the entire course.
    """
    
    return learner_prompt, instructor_prompt

def prompt_keyword_alignment(overlap_score, keyword_hits, total_keywords, top_keywords, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well key concepts are reinforced through course vocabulary. The **Keyword Alignment** metric reflects how consistently objective-level terminology appears in the transcript.

    **Evaluation Methodology:**
    - **Overlap Score:** Proportion of keywords from objectives found in the transcript (0–1).
    - **Keyword Hits:** Number of keywords matched, indicating concept coverage.
    - **Total Keywords:** Total terms derived from objectives for comparison.
    - **Final Rating:** Based on overlap, scaled from 1 (poor alignment) to 5 (excellent alignment).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Whether key learning concepts were repeatedly reinforced via vocabulary.

    **Prompt Instruction:**
    Given Overlap Score: {overlap_score}, Keyword Hits: {keyword_hits} out of {total_keywords}, Top Keywords: {top_keywords}, and Final Score: {final_rating}, write a short learner-style reflection on perceived concept reinforcement. Comment briefly on whether terminology felt consistently used and aligned with stated objectives.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Keyword Alignment**, which measures how accurately course vocabulary reflects stated learning objectives.

    **Methodology & Overview:**
    - Overlap Score reflects the proportion of objective terms present in the transcript (0 = none, 1 = complete).
    - Keyword Hits/Total validates precision of terminology usage.
    - Top Keywords lists terms inspected to ensure audit depth.

    **Final Score Rationale:**
    The final rating (1–5) is derived from overlap and hit counts, where 5 indicates thorough and accurate keyword usage, and 1 reflects major omissions.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Overlap Score: {overlap_score}, 
    Keyword Hits: {keyword_hits}/{total_keywords},  
    Top Keywords: {top_keywords}, 
    Final Rating: {final_rating}.  

    Please provide a 2–4 line summary stating whether coverage meets ideal expectations (overlap=1, all hits).  
    Suggestions should focus on improving missing or under-used vocabulary as needed.
    """

    return learner_prompt, instructor_prompt

def prompt_keyword_alignment_batch(overlap_scores, keyword_hits_list, total_keywords_list, top_keywords_list, final_ratings):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well **key concepts** were reinforced through course vocabulary across a batch of educational content. The **Keyword Alignment** metric captures how consistently important terminology appears throughout the transcripts.

    **Evaluation Methodology (across all samples):**
    - **Overlap Score (0–1):** Proportion of objective keywords actually appearing in the transcript.
    - **Keyword Hits vs Total Keywords:** Indicates depth of concept coverage.
    - **Final Rating (1–5):** Reflects overall alignment, with 5 representing strong reinforcement of key concepts.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence holistic summary.
    - Focus: Whether terminology felt consistently reinforced across the batch.

    **Prompt Instruction:**
    Given the following batch values:
    - Overlap Scores: {overlap_scores}
    - Keyword Hits: {keyword_hits_list}
    - Total Keywords: {total_keywords_list}
    - Top Keywords (per sample): {top_keywords_list}
    - Final Ratings: {final_ratings}

    Write a short learner-style reflection summarizing how effectively key learning concepts felt reinforced throughout the batch. Avoid metrics; focus on whether the terminology generally felt consistent and aligned with learning objectives.
    """

    instructor_prompt = f"""
    You are an AI assistant assessing **Keyword Alignment** across a batch of educational transcripts.

    **Evaluation Methodology (across all samples):**
    - **Overlap Score (0–1):** Proportion of objective keywords actually appearing in the transcript.
    - **Keyword Hits vs Total Keywords:** Indicates depth of concept coverage.
    - **Final Rating (1–5):** Reflects overall alignment, with 5 representing strong reinforcement of key concepts.

    **Metric Overview:**
    - **Overlap Scores (0–1):** {overlap_scores}
    - **Keyword Hits / Total Keywords:** {keyword_hits_list} / {total_keywords_list}
    - **Top Keywords (audit terms):** {top_keywords_list}
    - **Final Ratings (1–5):** {final_ratings}

    Provide a batch-level 2–4 line summary in passive voice.  
    Identify which aspect (overlap, hit count, or distribution of top keywords) deviates most from ideal expectations (overlap→1, all keywords covered).  
    Suggestions should be concise, formal, and focused on improving consistency and coverage of key learning terminology across the content.
    """
    return learner_prompt, instructor_prompt

def prompt_keyword_alignment_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how well **key concepts** were reinforced through course vocabulary across a **course** comprising multiple modules. The **Keyword Alignment** metric captures how consistently important terminology appears throughout the transcripts.

    **Evaluation Methodology:**
    - **Overlap Score (0–1):** Proportion of objective keywords actually appearing in the transcript.
    - **Keyword Hits vs Total Keywords:** Indicates depth of concept coverage.
    - **Final Rating (1–5):** Reflects overall alignment, with 5 representing strong reinforcement of key concepts.

    **Expected Response Guidelines:**
    - Perspective: Simulated learner’s point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence holistic reflection.
    - Focus: Whether terminology felt consistently reinforced throughout the course.

    **Prompt Instruction:**
    Given the following **module-level input**, reflect on the **overall course-level experience** in terms of concept reinforcement:

    {module_level_data}

    Write a short learner-style reflection summarizing how effectively the course reinforced its key learning concepts. Avoid citing metrics; focus on perceived terminology consistency and alignment with learning objectives.
    """

    instructor_prompt = f"""
    You are an AI assistant assessing **Keyword Alignment** across a course composed of multiple modules.

    **Evaluation Methodology:**
    - **Overlap Score (0–1):** Proportion of objective keywords actually appearing in the transcript.
    - **Keyword Hits vs Total Keywords:** Indicates depth of concept coverage.
    - **Final Rating (1–5):** Reflects overall alignment, with 5 representing strong reinforcement of key concepts.

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a 2–4 line instructor-style summary of **overall keyword alignment** across the course.
    - Identify which aspect (overlap, hit count, or distribution of top keywords) deviates most from ideal expectations (overlap→1, all keywords covered).
    - Provide a concise and actionable suggestion for improving consistency and coverage of learning terminology.
    - Avoid module-by-module analysis; focus on course-wide trends.

    Provide a high-level course-level assessment of keyword alignment quality.
    """

    return learner_prompt, instructor_prompt

def prompt_lo_coverage(lo_coverage_raw, covered_los, total_los, coverage_threshold, los_above_threshold, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of whether the course successfully delivers on its promised learning objectives. The **LO Coverage** metric reflects how completely those objectives are addressed across the content.

    **Evaluation Methodology:**
    - **LO Coverage Raw Score:** Fraction of objectives sufficiently matched (0 = none covered, 1 = fully covered).
    - **Covered Learning Objectives:** Number of objectives meeting the similarity threshold of {coverage_threshold:.2f}.
    - **Final Rating:** Scaled 1–5 score indicating completeness (1 = poor coverage, 5 = excellent alignment).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly in passive voice.
    - Focus: Whether the objectives were perceived as fully delivered.

    **Prompt Instruction:**
    Given LO Coverage Raw: {lo_coverage_raw}, Covered LOs: {covered_los}/{total_los}, Threshold: {coverage_threshold}, Covered Indices: {los_above_threshold}, and Final Score: {final_rating}, write a short learner-style reflection on content completeness. Reflect how well learning goals felt addressed.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating the **LO Coverage** of educational content, which measures how thoroughly learning objectives are addressed via semantic similarity.

    **Methodology & Overview:**
    - Raw score reflects the proportion of objectives exceeding the coverage threshold.
    - Objective counts reveal absolute coverage and pinpoint gaps in delivery.
    - Threshold ensures alignment with substantive learning outcomes, avoiding superficial hits.

    **Final Score Interpretation:**
    Final rating (1–5) is determined by overall objective coverage, where 5 indicates nearly all objectives are robustly supported and 1 signifies major instructional gaps.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    LO Coverage Raw Score: {lo_coverage_raw}, 
    Covered LOs: {covered_los}/{total_los}, 
    Threshold: {coverage_threshold}, 
    Well-Covered Objectives: {los_above_threshold}, 
    Final Rating: {final_rating}.

    Provide a 2–4 line statement highlighting objectives falling below threshold.  
    Suggestions should focus on improving coverage of missed objectives to enhance curriculum completeness.
    """

    return learner_prompt, instructor_prompt

def prompt_lo_coverage_batch(lo_coverage_raw_list, covered_los_list, total_los_list, coverage_threshold_list, los_above_threshold_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of whether a batch successfully delivers on their promised learning objectives. The **LO Coverage** metric reflects how completely those objectives are addressed across multiple learning experiences.

    **Evaluation Methodology:**
    - **Raw Coverage Score:** Fraction of objectives sufficiently matched (0 = none covered, 1 = fully covered).
    - **Covered Learning Objectives:** Number of objectives meeting the similarity threshold.
    - **Final Rating (1–5):** Indicates completeness and relevance of delivery.

    **Expected Response Guidelines:**
    - Perspective: Learner’s simulated viewpoint.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary.
    - Focus: Whether learning goals felt thoroughly delivered across the batch.

    **Prompt Instruction:**
    Given batch-level metrics:
    - Raw Coverage Scores: {lo_coverage_raw_list}
    - Covered LOs: {covered_los_list}/{total_los_list}
    - Thresholds: {coverage_threshold_list}
    - Indices Above Threshold: {los_above_threshold_list}
    - Final Ratings: {final_rating_list}

    Write a short learner-style reflection on how well learning objectives appeared to be covered across the entire batch. Avoid technical detail. Reflect on perceived completeness of coverage.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Learning Objective (LO) Coverage** across batch. This measures how thoroughly instructional content addresses promised outcomes.

    **Evaluation Methodology:**
    - **Raw Coverage Score:** Fraction of objectives sufficiently matched (0 = none covered, 1 = fully covered).
    - **Covered Learning Objectives:** Number of objectives meeting the similarity threshold.
    - **Final Rating (1–5):** Indicates completeness and relevance of delivery.

    **Batch Metric Overview:**
    - Raw Scores: {lo_coverage_raw_list}
    - Covered LOs: {covered_los_list}/{total_los_list}
    - Similarity Thresholds: {coverage_threshold_list}
    - Well-Covered Objectives: {los_above_threshold_list}
    - Final Ratings (1–5): {final_rating_list}

    **Instruction:**
    Provide a concise (2–4 line) batch-level summary identifying the objective-coverage component that deviates most from ideal (Raw Score → 1, Coverage Counts → Total).  
    Use passive voice and give actionable guidance on how he might improve completeness and alignment of learning objective coverage.
    """
    return learner_prompt, instructor_prompt

def prompt_lo_coverage_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of whether a **course** successfully delivers on its promised learning objectives across multiple modules. The **LO Coverage** metric reflects how completely those objectives are addressed.

    **Evaluation Methodology:**
    - **Raw Coverage Score:** Fraction of objectives sufficiently matched (0 = none covered, 1 = fully covered).
    - **Covered Learning Objectives:** Count of objectives meeting the similarity threshold.
    - **Final Rating (1–5):** Indicates completeness and relevance of delivery.

    **Expected Response Guidelines:**
    - Perspective: Learner’s simulated viewpoint.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary.
    - Focus: Whether learning goals felt thoroughly delivered throughout the course.

    **Prompt Instruction:**
    Given the following **module-level data**, reflect on the **overall course-level** perception of learning objective coverage:

    {module_level_data}

    Write a short learner-style reflection summarizing whether the course felt complete in delivering its stated objectives. Avoid technical breakdowns. Focus on perceived thoroughness and alignment.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Learning Objective (LO) Coverage** across a course composed of multiple modules. This measures how thoroughly instructional content addresses the promised outcomes.

    **Evaluation Methodology:**
    - **Raw Coverage Score:** Fraction of objectives sufficiently matched (0 = none covered, 1 = fully covered).
    - **Covered Learning Objectives:** Number of objectives meeting the similarity threshold.
    - **Final Rating (1–5):** Indicates completeness and relevance of delivery.

    **Course Module Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use formal tone and passive voice.
    - Write a 2–4 line **course-level** summary highlighting the most significant deviation from ideal coverage (ideal: Raw Score → 1, Covered LOs = Total LOs).
    - Suggest one actionable improvement to increase completeness and alignment.
    - Avoid module-by-module commentary — focus on patterns and trends across the course.

    Provide a concise instructor-style assessment summarizing coverage quality and opportunities for enhancement.
    """
    
    return learner_prompt, instructor_prompt

def prompt_alignment_balance(alignment_balance_raw, mean_similarity, std_similarity, coefficient_of_variation, similarity_range, final_rating):
    learner_prompt = f"""
    You are simulating a learner’s perception of how evenly the course addresses its stated objectives. The **Alignment Balance** metric reflects whether topics receive fair and proportional attention.

    **Evaluation Methodology:**
    - **Alignment Balance Raw Score (0–1):** Indicates coverage evenness; higher is better.
    - **Mean Similarity:** Central tendency of objective coverage.
    - **Standard Deviation / Coefficient of Variation:** Reflect spread and disparity.
    - **Similarity Range:** Highlights if any objectives are especially over- or under-emphasized.
    - **Final Rating:** Scaled 1 (imbalanced) to 5 (fair and comprehensive).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary, mostly passive voice.
    - Focus: Whether learning feels evenly distributed across objectives.

    **Prompt Instruction:**
    Given Alignment Balance Raw Score: {alignment_balance_raw}, Mean Similarity: {mean_similarity}, Standard Deviation: {std_similarity}, Coefficient of Variation: {coefficient_of_variation}, Similarity Range: {similarity_range}, and Final Score: {final_rating}, write a short learner-style reflection on topic balance. Reflect whether any objectives may receive noticeably more or less focus than expected.
    """

    instructor_prompt = f"""
`    You are an AI assistant evaluating the **Alignment Balance** metric, which reflects whether learning objectives are addressed evenly in the curriculum.

    **Methodology & Overview:**
    - Raw Balance Score (0–1): Higher indicates more even attention.
    - Mean Similarity, Standard Deviation, Coefficient of Variation: Together indicate dispersion in focus.
    - Similarity Range identifies any overly neglected or overemphasized objectives.

    **Final Score Rationale:**
    The final score (1–5) derives linearly, where 5 reflects perfectly balanced coverage and 1 signals major imbalance across objectives.

    **Feedback (passive voice, 2–4 lines, concise and improvement-focused):**
    Alignment Balance Raw Score: {alignment_balance_raw}, 
    Mean Similarity: {mean_similarity},  
    Std Dev: {std_similarity}, 
    Coefficient of Variation: {coefficient_of_variation},  
    Similarity Range: {similarity_range}, 
    Final Rating: {final_rating}.
    
    Please summarise in 2–4 lines which metric deviates most from ideal values, and recommend balance-oriented adjustments to elevate curriculum fairness.
    """

    return learner_prompt, instructor_prompt

def prompt_alignment_balance_batch(alignment_balance_raw_list, mean_similarity_list, std_similarity_list, coefficient_of_variation_list, similarity_range_list, final_rating_list):
    learner_prompt = f"""
    You are simulating a learner’s perception of how evenly a **batch** addresses their stated objectives. The **Alignment Balance** metric reflects whether topics receive fair and proportional attention.

    **Evaluation Methodology:**
    - **Alignment Balance Raw Score (0–1):** Indicates coverage evenness; higher is better.
    - **Mean Similarity:** Central tendency of objective coverage.
    - **Standard Deviation / Coefficient of Variation:** Reflect spread and disparity.
    - **Similarity Range:** Highlights if any objectives are especially over- or under-emphasized.
    - **Final Ratings:** Scaled 1 (imbalanced) to 5 (fair and comprehensive).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, concise.
    - Style: 2–4 sentence summary.
    - Focus: Whether learning *feels* evenly distributed across objectives overall.

    **Prompt Instruction:**
    Given batch-level values:
      • Raw Scores: {alignment_balance_raw_list}
      • Mean Similarities: {mean_similarity_list}
      • Std Deviations: {std_similarity_list}
      • Coefficient of Variations: {coefficient_of_variation_list}
      • Similarity Ranges: {similarity_range_list}
      • Final Ratings: {final_rating_list}

    Write a short learner-style reflection describing the **overall sense** of balance across objectives in this set.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Alignment Balance** for a batch.

    **Evaluation Methodology:**
    - **Alignment Balance Raw Score (0–1):** Indicates coverage evenness; higher is better.
    - **Mean Similarity:** Central tendency of objective coverage.
    - **Standard Deviation / Coefficient of Variation:** Reflect spread and disparity.
    - **Similarity Range:** Highlights if any objectives are especially over- or under-emphasized.
    - **Final Ratings:** Scaled 1 (imbalanced) to 5 (fair and comprehensive).

    **Metric Overview:**
      • Raw Scores: {alignment_balance_raw_list}
      • Mean Similarity: {mean_similarity_list}
      • Std Dev: {std_similarity_list}
      • Coefficient of Variation: {coefficient_of_variation_list}
      • Similarity Range: {similarity_range_list}
      • Final Ratings: {final_rating_list}

    **Instruction:**
    Provide a 2–4 line, passive-voice batch-level diagnostic. Identify which metric most consistently deviates from ideal balance and suggest improvements to enhance fairness of coverage.
    """
    return learner_prompt, instructor_prompt

def prompt_alignment_balance_course(module_level_data: str):
    learner_prompt = f"""
    You are simulating a learner’s perception of how evenly a **course** addresses its stated objectives across multiple modules. The **Alignment Balance** metric reflects whether topics receive fair and proportional attention throughout the learning journey.

    **Evaluation Methodology:**
    - **Alignment Balance Raw Score (0–1):** Indicates coverage evenness; higher values reflect better balance.
    - **Mean Similarity:** Central tendency of objective coverage.
    - **Standard Deviation / Coefficient of Variation:** Reflect spread and disparity in coverage.
    - **Similarity Range:** Shows if certain objectives were especially over- or under-emphasized.
    - **Final Ratings:** Scaled from 1 (imbalanced) to 5 (fair and comprehensive).

    **Expected Response Guidelines:**
    - Perspective: From a learner’s simulated point of view.
    - Tone: Formal, passive, and concise.
    - Style: 2–4 sentence reflection using passive voice wherever natural.
    - Focus: Whether the learner would *feel* that objectives were addressed evenly across the course.

    **Prompt Instruction:**
    Given the following **module-level data**:

    {module_level_data}

    Write a short learner-style reflection summarizing the **overall sense of balance** across objectives for the course. Avoid technical jargon and module-by-module commentary. Focus on the overall perception of fairness in content coverage.
    """

    instructor_prompt = f"""
    You are an AI assistant evaluating **Alignment Balance** across a course composed of multiple modules. This metric measures how evenly learning objectives are covered throughout the course.

    **Evaluation Methodology:**
    - **Alignment Balance Raw Score (0–1):** Higher is better.
    - **Mean Similarity:** Indicates typical alignment with objectives.
    - **Standard Deviation / Coefficient of Variation:** Highlight disparities in coverage.
    - **Similarity Range:** Reveals over- or under-emphasis of specific objectives.
    - **Final Ratings:** Scaled from 1 (imbalanced) to 5 (balanced).

    **Course Data:**
    {module_level_data}

    **Response Guidelines:**
    - Use passive voice and formal tone.
    - Write a 2–4 line, course-level summary identifying which metric most consistently deviated from ideal balance.
    - Provide one focused, actionable suggestion for improving fairness in coverage.
    - Avoid module-specific breakdowns — focus on general patterns.

    Provide a high-level diagnostic of the course’s alignment balance, emphasizing the most prevalent factor affecting fairness.
    """
    
    return learner_prompt, instructor_prompt

def analyze_transcript_alignment(los, transcript_text, file_name):
    """Analyze how well a transcript aligns with learning objectives using GPU acceleration"""
    if not los or not transcript_text:
        return {
            'Semantic Alignment Score': {'Value': 1, 'User Feedback': 'No content available for analysis', 'Instructor Feedback': 'No content available for analysis'},
            'Keyword Alignment Score': {'Value': 1, 'User Feedback': 'No content available for analysis', 'Instructor Feedback': 'No content available for analysis'},
            'LO Coverage Score': {'Value': 1, 'User Feedback': 'No content available for analysis', 'Instructor Feedback': 'No content available for analysis'},
            'Alignment Balance Score': {'Value': 1, 'User Feedback': 'No content available for analysis', 'Instructor Feedback': 'No content available for analysis'}
        }

    # Chunk the transcript
    chunks = chunk_transcript(transcript_text)
    if not chunks:
        return {
            'Semantic Alignment Score': {'Value': 1, 'User Feedback': 'No meaningful content chunks found', 'Instructor Feedback': 'No meaningful content chunks found'},
            'Keyword Alignment Score': {'Value': 1, 'User Feedback': 'No meaningful content chunks found', 'Instructor Feedback': 'No meaningful content chunks found'},
            'LO Coverage Score': {'Value': 1, 'User Feedback': 'No meaningful content chunks found', 'Instructor Feedback': 'No meaningful content chunks found'},
            'Alignment Balance Score': {'Value': 1, 'User Feedback': 'No meaningful content chunks found', 'Instructor Feedback': 'No meaningful content chunks found'}
        }

    # Calculate semantic similarity with GPU acceleration
    sim_matrix = calculate_semantic_similarity(los, chunks, t_model)

    # Use numpy vectorized operations for metrics calculation
    max_similarities = np.max(sim_matrix, axis=1)
    mean_similarities = np.mean(sim_matrix, axis=1)

    # 1. Semantic alignment: average of max similarities (0-1 range)
    semantic_alignment_raw = float(np.mean(max_similarities))
    semantic_alignment_score = round(float(np.clip(semantic_alignment_raw * 4 + 1, 1, 5)), 2)

    # 2. LO coverage: percentage of LOs with at least one good match
    threshold = 0.4  # Similarity threshold for "good" coverage
    covered_los = np.sum(max_similarities >= threshold)
    lo_coverage_raw = float(covered_los / len(los))
    lo_coverage_score = round(float(np.clip(lo_coverage_raw * 4 + 1, 1, 5)), 2)

    # 3. Alignment balance: how evenly the LOs are covered
    mean_sim = np.mean(max_similarities)
    std_sim = np.std(max_similarities)
    alignment_balance_raw = 1.0 - float(std_sim / (mean_sim + 1e-10))
    alignment_balance_raw = max(0.0, min(1.0, alignment_balance_raw))  # Clamp to [0,1]
    alignment_balance_score = round(float(np.clip(alignment_balance_raw * 4 + 1, 1, 5)), 2)
    coefficient_of_variation = std_sim / (mean_sim + 1e-10)
    similarity_range = {
        'min': round(float(np.min(max_similarities)), 3),
        'max': round(float(np.max(max_similarities)), 3)
    }

    # 4. Keyword alignment (already returns score and intermediates)
    keyword_result = calculate_keyword_overlap(los, transcript_text)

    # Generate user and instructor feedback using prompt functions
    semantic_score_user_feedback = execute_prompt(prompt_semantic_alignment(
        semantic_alignment_raw,
        [round(sim, 3) for sim in max_similarities],
        [round(sim, 3) for sim in mean_similarities],
        len(chunks),
        semantic_alignment_score
    )[0])

    semantic_score_instructor_feedback = execute_prompt(prompt_semantic_alignment(
        semantic_alignment_raw,
        [round(sim, 3) for sim in max_similarities],
        [round(sim, 3) for sim in mean_similarities],
        len(chunks),
        semantic_alignment_score
    )[1])

    keyword_score_user_feedback = execute_prompt(prompt_keyword_alignment(
        keyword_result['intermediates']['overlap_score'],
        keyword_result['intermediates']['keyword_hits'],
        keyword_result['intermediates']['total_keywords'],
        keyword_result['intermediates']['top_keywords'],
        keyword_result['score']
    )[0])

    keyword_score_instructor_feedback = execute_prompt(prompt_keyword_alignment(
        keyword_result['intermediates']['overlap_score'],
        keyword_result['intermediates']['keyword_hits'],
        keyword_result['intermediates']['total_keywords'],
        keyword_result['intermediates']['top_keywords'],
        keyword_result['score']
    )[1])

    coverage_score_user_feedback = execute_prompt(prompt_lo_coverage(
        lo_coverage_raw,
        int(covered_los),
        len(los),
        threshold,
        [i for i, sim in enumerate(max_similarities) if sim >= threshold],
        lo_coverage_score
    )[0])

    coverage_score_instructor_feedback = execute_prompt(prompt_lo_coverage(
        lo_coverage_raw,
        int(covered_los),
        len(los),
        threshold,
        [i for i, sim in enumerate(max_similarities) if sim >= threshold],
        lo_coverage_score
    )[1])

    balance_score_user_feedback = execute_prompt(prompt_alignment_balance(
        alignment_balance_raw,
        round(mean_sim, 3),
        round(std_sim, 3),
        round(coefficient_of_variation, 3),
        similarity_range,
        alignment_balance_score
    )[0])

    balance_score_instructor_feedback = execute_prompt(prompt_alignment_balance(
        alignment_balance_raw,
        round(mean_sim, 3),
        round(std_sim, 3),
        round(coefficient_of_variation, 3),
        similarity_range,
        alignment_balance_score
    )[1])

    return {
        "File Name": file_name,
        "Semantic Alignment Score": {
            "Intermediate Parameters": {
                "Raw Score": semantic_alignment_raw,
                "Max Similarities": [round(sim, 3) for sim in max_similarities],
                "Mean Similarities": [round(sim, 3) for sim in mean_similarities],
                "No.of Chunks": len(chunks),
            },
            "Final Score": semantic_alignment_score,
            "User Perspective Assessment": semantic_score_user_feedback,
            "Instructor Feedback": semantic_score_instructor_feedback
        },
        "Keyword Alignment Score": {
            "Intermediate Parameters": {
                "Raw Score": keyword_result['intermediates']['overlap_score'],
                "Keyword Hits": keyword_result['intermediates']['keyword_hits'],
                "Total Keywords": keyword_result['intermediates']['total_keywords'],
                "Top Keywords": keyword_result['intermediates']['top_keywords']
            },
            "Final Score": keyword_result['score'],
            "User Perspective Assessment": keyword_score_user_feedback,
            "Instructor Feedback": keyword_score_instructor_feedback
        },
        "LO Coverage Score": {
            "Intermediate Parameters": {
                "Raw Score": lo_coverage_raw,
                "Covered LOs": int(covered_los),
                "Total LOs": len(los),
                "Threshold": threshold,
                "Above Threshold LOs": [i for i, sim in enumerate(max_similarities) if sim >= threshold]
            },
            "Final Score": lo_coverage_score,
            "User Perspective Assessment": coverage_score_user_feedback,
            "Instructor Feedback": coverage_score_instructor_feedback
        },
        "Alignment Balance Score": {
            "Intermediate Parameters": {
                "Raw Score": alignment_balance_raw,
                "Mean Similarity": round(mean_sim, 3),
                "Std Dev Similarity": round(std_sim, 3),
                "Coefficient of Variation": round(coefficient_of_variation, 3),
                "Similarity Range": similarity_range
            },
            "Final Score": alignment_balance_score,
            "User Perspective Assessment": balance_score_user_feedback,
            "Instructor Feedback": balance_score_instructor_feedback
        }
    }

def analyze_module(module_name, los, transcript_files, service_account_file, reading_contents):
    """Analyze all transcripts for a module"""
    print(f"\nAnalyzing module: {module_name}")
    print(f"Found {len(transcript_files)} transcript files")
    print(f"Found {len(reading_contents)} reading files")
    print(f"Learning Objectives: {len(los)}")
    transcript_metrics = Parallel(n_jobs=os.cpu_count()-1)(
        delayed(analyze_transcript_alignment)(
            los,
            read_transcript(file_path, service_account_file),
            file_name
        ) for file_path, file_name in tqdm(transcript_files, desc="Processing transcripts")
    )
    reading_metrics = [analyze_transcript_alignment(los, content, file_name) for file_name, content in reading_contents]
    module_metrics = transcript_metrics + reading_metrics
    return module_metrics

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
                      if each_file["extension"] == ".txt":
                        file_data["txt_file"] = each_file["id"]
                        file_data["file_name"] = each_file["directory"].split("/")[-1]
                      if each_file["extension"] == ".pdf":
                        file_data["pdf_file"] = each_file["id"]
                  module_files[module_name].append(file_data)
    return module_files

def save_dict_to_json(data: dict, file_path: str):
    """
    Saves a dictionary to a JSON file.

    Parameters:
    - data: dict, the dictionary to save
    - file_path: str, path to the output .json file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

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

def perform_subtitle_analysis(service_account_file, folder_id):
    result_1 = {}
    result_2 = {}
    result_1["Course Name"] = get_folder_name(folder_id, service_account_file)
    result_1["Course Name"] = get_folder_name(folder_id, service_account_file)
    service = get_gdrive_service(service_account_file)
    # Try to fetch metadata.json
    metadata = fetch_metadata_json(service, folder_id)
    extensions = ['.txt', '.pdf']
    all_files = find_files_recursively(service, folder_id, extensions)
    #print(all_files)
    module_files = organize_files_by_module(metadata, all_files)
    transcript_results = []
    total_results_1 = []
    total_results_2 = []
    count = 0
    for key, value in metadata.items():
        if count >= 2:
            break
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            module_los = value.get("Learning Objectives", [])
            module_results_1 = {}
            module_results_1["Name"] = module_name
            module_results_1["Results"] = []
            module_results_2 = {}
            module_results_2["Name"] = module_name
            module_results_2["Results"] = []
            if not module_los:
                print(f"Warning: No learning objectives found for {module_name}. Skipping.")
                continue
            # Get transcript files for this module
            transcript_files = [(d["txt_file"],d["file_name"]) for d in module_files[module_name] if "txt_file" in d][:2]
            print(f"Transcript Files: {len(transcript_files)}")
            reading_files = [d["pdf_file"] for d in module_files[module_name] if "pdf_file" in d][:1]
            custom_extraction_prompt = None
            reading_contents = [read_reading(file_path, service_account_file, custom_extraction_prompt) for file_path in reading_files]
            transcript_results_temp = Parallel(n_jobs=os.cpu_count() - 1, backend="multiprocessing")(
                delayed(evaluate_transcript_content)(file_path, service_account_file, file_name)
                for file_path, file_name in tqdm(transcript_files)
                )
            reading_results_temp = [evaluate_reading_content(file_name, reading_content) for file_name, reading_content in reading_contents]
            transcript_results.extend(transcript_results_temp + reading_results_temp)
            module_results_1["Results"] = transcript_results_temp + reading_results_temp
            module_results_1["Easy-to-Understand Score"] = {}
            easy_to_understand_ratings = [each["Easy-to-Understand Score"]["Final Score"] for each in module_results_1["Results"]]
            avg_word_length_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Average Word Length"] for each in module_results_1["Results"]]
            avg_syllables_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Average Syllables per word"] for each in module_results_1["Results"]]
            raw_scores_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_1["Results"]]
            module_results_1["Easy-to-Understand Score"]["Score"] = round(sum(easy_to_understand_ratings)/len(easy_to_understand_ratings), 2)
            learner_prompt, instructor_prompt = prompt_easy_to_understand_batch(avg_word_length_list, avg_syllables_list, raw_scores_list, easy_to_understand_ratings)
            module_results_1["Easy-to-Understand Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_1["Easy-to-Understand Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_1["Engagement Score"] = {}
            engagement_ratings = [each["Engagement Score"]["Final Score"] for each in module_results_1["Results"]]
            avg_sentiment_list = [each["Engagement Score"]["Intermediate Parameters"]["Average Sentiment"] for each in module_results_1["Results"]]
            positive_ratio_list = [each["Engagement Score"]["Intermediate Parameters"]["Positive Ratio"] for each in module_results_1["Results"]]
            raw_scores_list = [each["Engagement Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_1["Results"]]
            module_results_1["Engagement Score"]["Score"] = round(sum(engagement_ratings)/len(engagement_ratings), 2)
            learner_prompt, instructor_prompt = prompt_engagement_batch(avg_sentiment_list, positive_ratio_list, raw_scores_list, engagement_ratings)
            module_results_1["Engagement Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_1["Engagement Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_1["Pacing & Flow Score"] = {}
            pacing_n_flow_ratings = [each["Pacing & Flow Score"]["Final Score"] for each in module_results_1["Results"]]
            avg_similarity_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Average Similarity"] for each in module_results_1["Results"]]
            similarity_std_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Similarity Std Dev"] for each in module_results_1["Results"]]
            consistency_score_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Consistency Score"] for each in module_results_1["Results"]]
            raw_scores_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_1["Results"]]
            module_results_1["Pacing & Flow Score"]["Score"] = round(sum(pacing_n_flow_ratings)/len(pacing_n_flow_ratings), 2)
            learner_prompt, instructor_prompt = prompt_pacing_flow_batch(avg_similarity_list, similarity_std_list, consistency_score_list, raw_scores_list, pacing_n_flow_ratings)
            module_results_1["Pacing & Flow Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_1["Pacing & Flow Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_1["Well Structured Score"] = {}
            well_structured_ratings = [each["Well Structured Score"]["Final Score"] for each in module_results_1["Results"]]
            unique_clusters_list = [each["Well Structured Score"]["Intermediate Parameters"]["Unique Clusters"] for each in module_results_1["Results"]]
            total_clusters_list = [each["Well Structured Score"]["Intermediate Parameters"]["Total Clusters"] for each in module_results_1["Results"]]
            cluster_diversity_list = [each["Well Structured Score"]["Intermediate Parameters"]["Cluster Diversity"] for each in module_results_1["Results"]]
            raw_scores_list = [each["Well Structured Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_1["Results"]]
            module_results_1["Well Structured Score"]["Score"] = round(sum(well_structured_ratings)/len(well_structured_ratings), 2)
            learner_prompt, instructor_prompt = prompt_well_structured_batch(unique_clusters_list, total_clusters_list, cluster_diversity_list, raw_scores_list, well_structured_ratings)
            module_results_1["Well Structured Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_1["Well Structured Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_1["Learning Value Score"] = {}
            learning_value_ratings = [each["Learning Value Score"]["Final Score"] for each in module_results_1["Results"]]
            avg_coherence_list = [each["Learning Value Score"]["Intermediate Parameters"]["Average Coherence"] for each in module_results_1["Results"]]
            coherence_std_list = [each["Learning Value Score"]["Intermediate Parameters"]["Coherence Std Dev"] for each in module_results_1["Results"]]
            focus_consistency_list = [each["Learning Value Score"]["Intermediate Parameters"]["Focus Consistency"] for each in module_results_1["Results"]]
            raw_scores_list = [each["Learning Value Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_1["Results"]]
            module_results_1["Learning Value Score"]["Score"] = round(sum(learning_value_ratings)/len(learning_value_ratings), 2)
            learner_prompt, instructor_prompt = prompt_learning_value_batch(avg_coherence_list, coherence_std_list, focus_consistency_list, raw_scores_list, learning_value_ratings)
            module_results_1["Learning Value Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_1["Learning Value Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            if not transcript_files:
                print(f"Warning: No transcript files found for {module_name}. Skipping.")
                continue
            # Analyze the module
            module_metrics = analyze_module(module_name, module_los, transcript_files, service_account_file, reading_contents)
            module_results_2["Results"] = module_metrics
            module_results_2["Semantic Alignment Score"] = {}
            semantic_alignment_ratings = [each["Semantic Alignment Score"]["Final Score"] for each in module_results_2["Results"]]
            raw_scores_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_2["Results"]]
            max_similarities_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Max Similarities"] for each in module_results_2["Results"]]
            mean_similarities_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Mean Similarities"] for each in module_results_2["Results"]]
            num_chunks_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["No.of Chunks"] for each in module_results_2["Results"]]
            module_results_2["Semantic Alignment Score"]["Score"] = round(sum(semantic_alignment_ratings)/len(semantic_alignment_ratings), 2)
            learner_prompt, instructor_prompt = prompt_semantic_alignment_batch(raw_scores_list, max_similarities_list, mean_similarities_list, num_chunks_list, semantic_alignment_ratings)
            module_results_2["Semantic Alignment Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_2["Semantic Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_2["Keyword Alignment Score"] = {}
            keyword_alignment_ratings = [each["Keyword Alignment Score"]["Final Score"] for each in module_results_2["Results"]]
            raw_scores_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_2["Results"]]
            keyword_hits_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Keyword Hits"] for each in module_results_2["Results"]]
            total_keywords_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Total Keywords"] for each in module_results_2["Results"]]
            top_keywords_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Top Keywords"] for each in module_results_2["Results"]]
            module_results_2["Keyword Alignment Score"]["Score"] = round(sum(keyword_alignment_ratings)/len(keyword_alignment_ratings), 2)
            learner_prompt, instructor_prompt = prompt_keyword_alignment_batch(raw_scores_list, keyword_hits_list, total_keywords_list, top_keywords_list, keyword_alignment_ratings)
            module_results_2["Keyword Alignment Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_2["Keyword Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_2["LO Coverage Score"] = {}
            lo_coverage_ratings = [each["LO Coverage Score"]["Final Score"] for each in module_results_2["Results"]]
            raw_scores_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_2["Results"]]
            covered_los_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Covered LOs"] for each in module_results_2["Results"]]
            total_los_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Total LOs"] for each in module_results_2["Results"]]
            threshold_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Threshold"] for each in module_results_2["Results"]]
            above_threshold_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Above Threshold LOs"] for each in module_results_2["Results"]]
            module_results_2["LO Coverage Score"]["Score"] = round(sum(lo_coverage_ratings)/len(lo_coverage_ratings), 2)
            learner_prompt, instructor_prompt = prompt_lo_coverage_batch(raw_scores_list, covered_los_list, total_los_list, threshold_list, above_threshold_list, lo_coverage_ratings)
            module_results_2["LO Coverage Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_2["LO Coverage Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            module_results_2["Alignment Balance Score"] = {}
            alignment_balance_ratings = [each["Alignment Balance Score"]["Final Score"] for each in module_results_2["Results"]]
            raw_scores_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Raw Score"] for each in module_results_2["Results"]]
            mean_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Mean Similarity"] for each in module_results_2["Results"]]
            std_dev_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Std Dev Similarity"] for each in module_results_2["Results"]]
            cv_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Coefficient of Variation"] for each in module_results_2["Results"]]
            similarity_ranges_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Similarity Range"] for each in module_results_2["Results"]]
            module_results_2["Alignment Balance Score"]["Score"] = round(sum(alignment_balance_ratings)/len(alignment_balance_ratings), 2)
            learner_prompt, instructor_prompt = prompt_alignment_balance_batch(raw_scores_list, mean_similarities_list, std_dev_similarities_list, cv_similarities_list, similarity_ranges_list, alignment_balance_ratings)
            module_results_2["Alignment Balance Score"]["User Perspective Assessment"] = execute_prompt(learner_prompt)
            module_results_2["Alignment Balance Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
            total_results_1.append(module_results_1)
            total_results_2.append(module_results_2)
            count += 1
    result_1["Module Results"] = total_results_1
    ###########################################################################################################
    result_1["Course Results"] = {}
    result_1["Course Results"]["Easy-to-Understand Score"] = {}
    result_1["Course Results"]["Engagement Score"] = {}
    result_1["Course Results"]["Pacing & Flow Score"] = {}
    result_1["Course Results"]["Well Structured Score"] = {}
    result_1["Course Results"]["Learning Value Score"] = {}
    easy_2_understand_parameter = ""
    engagement_parameter = ""
    pacing_and_flow_parameter = ""
    well_structured_parameter = ""
    learning_value_parameter = ""
    final_easy_to_understand_scores = []
    final_pacing_and_flow_scores = []
    final_engagement_scores = []
    final_well_structured_scores = []
    final_learning_value_scores = []
    for module_result in result_1["Module Results"]:
        module_name = module_result["Name"]
        easy_to_understand_ratings = [each["Easy-to-Understand Score"]["Final Score"] for each in module_result["Results"]]
        avg_word_length_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Average Word Length"] for each in module_result["Results"]]
        avg_syllables_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Average Syllables per word"] for each in module_result["Results"]]
        raw_scores_list = [each["Easy-to-Understand Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        easy_2_understand_parameter += (
            f"Module Name: {module_name}\n"
            f"Average Word Length List: {avg_word_length_list}\n"
            f"Average Syllables List: {avg_syllables_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Easy to Understand Ratings: {easy_to_understand_ratings}\n"
        )
        final_easy_to_understand_scores.append(module_result["Easy-to-Understand Score"]["Score"])
        engagement_ratings = [each["Engagement Score"]["Final Score"] for each in module_result["Results"]]
        avg_sentiment_list = [each["Engagement Score"]["Intermediate Parameters"]["Average Sentiment"] for each in module_result["Results"]]
        positive_ratio_list = [each["Engagement Score"]["Intermediate Parameters"]["Positive Ratio"] for each in module_result["Results"]]
        raw_scores_list = [each["Engagement Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        engagement_parameter += (
            f"Module Name: {module_name}\n"
            f"Average Sentiment List: {avg_sentiment_list}\n"
            f"Positive Ratio List: {positive_ratio_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Engagement Ratings: {engagement_ratings}\n"
        )
        final_engagement_scores.append(module_result["Engagement Score"]["Score"])
        pacing_n_flow_ratings = [each["Pacing & Flow Score"]["Final Score"] for each in module_result["Results"]]
        avg_similarity_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Average Similarity"] for each in module_result["Results"]]
        similarity_std_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Similarity Std Dev"] for each in module_result["Results"]]
        consistency_score_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Consistency Score"] for each in module_result["Results"]]
        raw_scores_list = [each["Pacing & Flow Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        pacing_and_flow_parameter += (
            f"Module Name: {module_name}\n"
            f"Average Similarties List: {avg_similarity_list}\n"
            f"Similarity Std Dev List: {similarity_std_list}\n"
            f"Consistency Scores List: {consistency_score_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Pacing and Flow Ratings: {pacing_n_flow_ratings}\n"
        )
        final_pacing_and_flow_scores.append(module_result["Pacing & Flow Score"]["Score"])
        well_structured_ratings = [each["Well Structured Score"]["Final Score"] for each in module_result["Results"]]
        unique_clusters_list = [each["Well Structured Score"]["Intermediate Parameters"]["Unique Clusters"] for each in module_result["Results"]]
        total_clusters_list = [each["Well Structured Score"]["Intermediate Parameters"]["Total Clusters"] for each in module_result["Results"]]
        cluster_diversity_list = [each["Well Structured Score"]["Intermediate Parameters"]["Cluster Diversity"] for each in module_result["Results"]]
        raw_scores_list = [each["Well Structured Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        well_structured_parameter += (
            f"Module Name: {module_name}\n"
            f"Unique Clusters List: {unique_clusters_list}\n"
            f"Total Clusters List: {total_clusters_list}\n"
            f"Cluster Diversity List: {cluster_diversity_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Well Structured Ratings: {well_structured_ratings}\n"
        )
        final_well_structured_scores.append(module_result["Well Structured Score"]["Score"])
        learning_value_ratings = [each["Learning Value Score"]["Final Score"] for each in module_result["Results"]]
        avg_coherence_list = [each["Learning Value Score"]["Intermediate Parameters"]["Average Coherence"] for each in module_result["Results"]]
        coherence_std_list = [each["Learning Value Score"]["Intermediate Parameters"]["Coherence Std Dev"] for each in module_result["Results"]]
        focus_consistency_list = [each["Learning Value Score"]["Intermediate Parameters"]["Focus Consistency"] for each in module_result["Results"]]
        raw_scores_list = [each["Learning Value Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        learning_value_parameter += (
            f"Module Name: {module_name}\n"
            f"Average Coherence List: {avg_coherence_list}\n"
            f"Coherence Std Dev List: {coherence_std_list}\n"
            f"Focus Consistency List: {focus_consistency_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Learning Value Ratings: {learning_value_ratings}\n"
        )
        final_learning_value_scores.append(module_result["Learning Value Score"]["Score"])
    learner_prompt, instructor_prompt = prompt_easy_to_understand_course(easy_2_understand_parameter)
    result_1["Course Results"]["Easy-to-Understand Score"]["Score"] = round(sum(final_easy_to_understand_scores)/len(final_easy_to_understand_scores),2)
    result_1["Course Results"]["Easy-to-Understand Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_1["Course Results"]["Easy-to-Understand Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_engagement_course(engagement_parameter)
    result_1["Course Results"]["Engagement Score"]["Score"] = round(sum(final_engagement_scores)/len(final_engagement_scores),2)
    result_1["Course Results"]["Engagement Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_1["Course Results"]["Engagement Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_pacing_flow_course(pacing_and_flow_parameter)
    result_1["Course Results"]["Pacing & Flow Score"]["Score"] = round(sum(final_pacing_and_flow_scores)/len(final_pacing_and_flow_scores),2)
    result_1["Course Results"]["Pacing & Flow Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_1["Course Results"]["Pacing & Flow Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_well_structured_course(well_structured_parameter)
    result_1["Course Results"]["Well Structured Score"]["Score"] = round(sum(final_well_structured_scores)/len(final_well_structured_scores),2)
    result_1["Course Results"]["Well Structured Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_1["Course Results"]["Well Structured Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_learning_value_course(learning_value_parameter)
    result_1["Course Results"]["Learning Value Score"]["Score"] = round(sum(final_learning_value_scores)/len(final_learning_value_scores),2)
    result_1["Course Results"]["Learning Value Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_1["Course Results"]["Learning Value Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    ###########################################################################################################
    result_2["Module Results"] = total_results_2
    result_2["Course Results"] = {}
    result_2["Course Results"]["Semantic Alignment Score"] = {}
    result_2["Course Results"]["Keyword Alignment Score"] = {}
    result_2["Course Results"]["LO Coverage Score"] = {}
    result_2["Course Results"]["Alignment Balance Score"] = {}
    semantic_alignment_parameter = ""
    keyword_alignment_parameter = ""
    lo_coverage_parameter = ""
    alignment_balance_parameter = ""
    final_semantic_alignment_scores = []
    final_keyword_alignment_scores = []
    final_lo_coverage_scores = []
    final_alignment_balance_scores = []
    for module_result in result_2["Module Results"]:
        module_name = module_result["Name"]
        semantic_alignment_ratings = [each["Semantic Alignment Score"]["Final Score"] for each in module_result["Results"]]
        raw_scores_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        max_similarities_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Max Similarities"] for each in module_result["Results"]]
        mean_similarities_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["Mean Similarities"] for each in module_result["Results"]]
        num_chunks_list = [each["Semantic Alignment Score"]["Intermediate Parameters"]["No.of Chunks"] for each in module_result["Results"]]
        semantic_alignment_parameter += (
            f"Module Name: {module_name}\n"
            f"Max Similarities List: {max_similarities_list}\n"
            f"Mean Similarities List: {mean_similarities_list}\n"
            f"Num Chunks List: {num_chunks_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Semantic Alignment Ratings: {semantic_alignment_ratings}\n"
        )
        final_semantic_alignment_scores.append(module_result["Semantic Alignment Score"]["Score"])
        keyword_alignment_ratings = [each["Keyword Alignment Score"]["Final Score"] for each in module_result["Results"]]
        raw_scores_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        keyword_hits_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Keyword Hits"] for each in module_result["Results"]]
        total_keywords_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Total Keywords"] for each in module_result["Results"]]
        top_keywords_list = [each["Keyword Alignment Score"]["Intermediate Parameters"]["Top Keywords"] for each in module_result["Results"]]
        keyword_alignment_parameter += (
            f"Module Name: {module_name}\n"
            f"Overlap Scores List: {raw_scores_list}\n"
            f"Keyword Hits List {keyword_hits_list}\n"
            f"Total Keywords List: {total_keywords_list}\n"
            f"Top Keywords List: {top_keywords_list}\n"
            f"Final Keyword Alignment Ratings: {keyword_alignment_ratings}\n"
        )
        final_keyword_alignment_scores.append(module_result["Keyword Alignment Score"]["Score"])
        lo_coverage_ratings = [each["LO Coverage Score"]["Final Score"] for each in module_result["Results"]]
        raw_scores_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        covered_los_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Covered LOs"] for each in module_result["Results"]]
        total_los_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Total LOs"] for each in module_result["Results"]]
        threshold_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Threshold"] for each in module_result["Results"]]
        above_threshold_list = [each["LO Coverage Score"]["Intermediate Parameters"]["Above Threshold LOs"] for each in module_result["Results"]]
        lo_coverage_parameter += (
            f"Module Name: {module_name}\n"
            f"Covered LOs List: {covered_los_list}\n"
            f"Total LOs List {total_los_list}\n"
            f"Thresold List: {threshold_list}\n"
            f"Above Threshold LOs List: {above_threshold_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Lo Coverage Ratings: {lo_coverage_ratings}\n"
        )
        final_lo_coverage_scores.append(module_result["LO Coverage Score"]["Score"])
        alignment_balance_ratings = [each["Alignment Balance Score"]["Final Score"] for each in module_result["Results"]]
        raw_scores_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Raw Score"] for each in module_result["Results"]]
        mean_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Mean Similarity"] for each in module_result["Results"]]
        std_dev_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Std Dev Similarity"] for each in module_result["Results"]]
        cv_similarities_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Coefficient of Variation"] for each in module_result["Results"]]
        similarity_ranges_list = [each["Alignment Balance Score"]["Intermediate Parameters"]["Similarity Range"] for each in module_result["Results"]]
        alignment_balance_parameter += (
            f"Module Name: {module_name}\n"
            f"Mean Similarities List: {mean_similarities_list}\n"
            f"Std Dev Similarities List: {std_dev_similarities_list}\n"
            f"Coef of Var Similarities List: {cv_similarities_list}\n"
            f"Similarity Ranges List: {similarity_ranges_list}\n"
            f"Raw Scores List: {raw_scores_list}\n"
            f"Final Alignment Balance Ratings: {alignment_balance_ratings}\n"
        )
        final_alignment_balance_scores.append(module_result["Alignment Balance Score"]["Score"])
    learner_prompt, instructor_prompt = prompt_semantic_alignment_course(semantic_alignment_parameter)
    result_2["Course Results"]["Semantic Alignment Score"]["Score"] = round(sum(final_semantic_alignment_scores)/len(final_semantic_alignment_scores),2)
    result_2["Course Results"]["Semantic Alignment Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_2["Course Results"]["Semantic Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_keyword_alignment_course(keyword_alignment_parameter)
    result_2["Course Results"]["Keyword Alignment Score"]["Score"] = round(sum(final_keyword_alignment_scores)/len(final_keyword_alignment_scores),2)
    result_2["Course Results"]["Keyword Alignment Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_2["Course Results"]["Keyword Alignment Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_lo_coverage_course(lo_coverage_parameter)
    result_2["Course Results"]["LO Coverage Score"]["Score"] = round(sum(final_lo_coverage_scores)/len(final_lo_coverage_scores),2)
    result_2["Course Results"]["LO Coverage Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_2["Course Results"]["LO Coverage Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    learner_prompt, instructor_prompt = prompt_alignment_balance_course(alignment_balance_parameter)
    result_2["Course Results"]["Alignment Balance Score"]["Score"] = round(sum(final_alignment_balance_scores)/len(final_alignment_balance_scores),2)
    result_2["Course Results"]["Alignment Balance Score"]["Learner Perspective Assessment"] = execute_prompt(learner_prompt)
    result_2["Course Results"]["Alignment Balance Score"]["Instructor Feedback"] = execute_prompt(instructor_prompt)
    save_dict_to_json(convert_to_serializable(result_1), "Text Quality Results.json")
    save_dict_to_json(convert_to_serializable(result_2), "Text LO Validation Results.json")
    print(f"Analysis complete!")

if __name__ == "__main__":
    start_time = time.time()
    service_account_file = ""
    folder_id = ""
    perform_subtitle_analysis(service_account_file, folder_id)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Total analysis time: {elapsed_minutes:.2f} minutes")
