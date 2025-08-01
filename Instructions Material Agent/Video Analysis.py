from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

# ✅ Colab-ready VideoLOAnalyzer.py

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
import json
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
import deepface as DeepFace
import torch
import os
import re
import webvtt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import google.generativeai as genai

class VideoLOAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_keyframes(self, video_path, interval=5):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)

        keyframes = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append(frame_rgb)
            count += 1
        cap.release()
        return keyframes

    def encode_video_frames(self, keyframes, batch_size=8):
        processed_frames = self.processor(images=keyframes, return_tensors="pt", padding=True)
        frame_features = []

        for i in range(0, len(keyframes), batch_size):
            batch = {k: v[i:i+batch_size].to(self.device) for k, v in processed_frames.items()}
            with torch.no_grad():
                features = self.model.get_image_features(**batch)
            frame_features.append(features.cpu())

        return torch.cat(frame_features)

    def encode_learning_objectives(self, learning_objectives):
        inputs = self.processor(text=learning_objectives, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu()

    def calculate_alignment(self, frame_features, text_features):
        frame_features /= frame_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = frame_features @ text_features.T
        return similarity.numpy()

    def analyze_module(self, video_path, learning_objectives, interval=5):
        keyframes = self.extract_keyframes(video_path, interval)
        frame_features = self.encode_video_frames(keyframes)
        lo_features = self.encode_learning_objectives(learning_objectives)
        similarity_matrix = self.calculate_alignment(frame_features, lo_features)

        # 🟢 Compute raw metrics first
        per_lo_scores = self._calculate_per_lo_scores(similarity_matrix)
        temporal_coverage = self._calculate_temporal_coverage(similarity_matrix)
        concept_density = self._calculate_concept_density(similarity_matrix)

        max_coverage = max(temporal_coverage) if temporal_coverage else 0
        if max_coverage > 0:
            normalized_temporal_coverage = [float(x) / max_coverage for x in temporal_coverage]
        else:
            normalized_temporal_coverage = [0.0 for _ in temporal_coverage]

        per_lo_scores_scaled = {lo: round(float(1 + 4 * s), 4) for lo, s in zip(learning_objectives, per_lo_scores)}
        temporal_coverage_scaled = {lo: round(float(1 + 4 * s), 4) for lo, s in zip(learning_objectives, normalized_temporal_coverage)}
        concept_density_scaled = {lo: round(float(1 + 4 * s), 4) for lo, s in zip(learning_objectives, concept_density)}

        visual_consistency = float(self._calculate_visual_consistency(frame_features))
        visual_consistency_score = round(visual_consistency, 4)  # already 1–5 scaled

        metrics = {
            'per_lo_scores_scaled_1_to_5': per_lo_scores_scaled,
            'temporal_coverage_scaled_1_to_5': temporal_coverage_scaled,
            'concept_density_scaled_1_to_5': concept_density_scaled,
            'visual_consistency_scaled_1_to_5': visual_consistency_score
        }

        return metrics, similarity_matrix, keyframes


    def _calculate_per_lo_scores(self, similarity_matrix):
      scores = []
      for i in range(similarity_matrix.shape[1]):
          lo_scores = similarity_matrix[:, i]
          weights = np.clip((lo_scores - 0.2) / 0.6, 0, 1)  # range 0–1 scaled
          weighted = np.average(weights)
          scores.append(weighted)
      return np.array(scores)

    def _calculate_temporal_coverage(self, similarity_matrix, threshold=0.3):
      frame_count = similarity_matrix.shape[0]
      return [
          np.sum(similarity_matrix[:, i] > threshold) / frame_count
          for i in range(similarity_matrix.shape[1])
      ]

    def _calculate_concept_density(self, similarity_matrix, top_k=10):
      top_k_avg = []
      for col in similarity_matrix.T:  # iterate over LOs
          top_k_scores = np.sort(col)[-top_k:]
          top_k_avg.append(np.mean(top_k_scores))
      return np.array(top_k_avg)


    def _calculate_visual_consistency(self, frame_features, n_clusters=5):
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(frame_features.numpy())
      counts = np.bincount(kmeans.labels_)
      dominant_cluster_ratio = counts.max() / len(frame_features)
      return round(1 + 4 * dominant_cluster_ratio, 3)  # scaled to 1–5

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Colab path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = YOLO('yolov8n.pt').to(DEVICE)
FACE_DETECTOR_BACKEND = 'mtcnn'  # upgraded face detection

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

# ----------------------------
# ✅ SUBTITLE HELPERS
# ----------------------------

def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    return ' '.join([t for t in tokens if t.isalnum() and t not in stop_words])

def parse_subtitles(vtt_path):
    subtitles = []
    for caption in webvtt.read(vtt_path):
        subtitles.append({
            'start': caption.start_in_seconds,
            'end': caption.end_in_seconds,
            'text': preprocess_text(caption.text)
        })
    return subtitles

def calculate_subtitle_alignment(subtitles, visual_texts, slide_change_times):
    subtitle_texts = [s['text'] for s in subtitles]
    visual_texts_combined = [' '.join(visual_texts)]
    subtitle_texts_combined = [' '.join(subtitle_texts)]

    vectorizer = TfidfVectorizer().fit(visual_texts_combined + subtitle_texts_combined)
    tfidf_matrix = vectorizer.transform(visual_texts_combined + subtitle_texts_combined)
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    subtitle_times = [s['start'] for s in subtitles]
    matched_transitions = sum([1 for t in slide_change_times if any(abs(t - st) < 1.5 for st in subtitle_times)])
    temporal_alignment = matched_transitions / len(slide_change_times) if slide_change_times else 0

    subtitle_keywords = set(' '.join(subtitle_texts).split())
    visual_keywords = set(' '.join(visual_texts).split())
    concept_overlap = compute_conceptual_alignment(subtitle_texts, visual_texts)

    visual_refs = sum(['see' in s['text'] or 'shown' in s['text'] or 'diagram' in s['text'] for s in subtitles])
    matched_refs = sum(['see' in s['text'] and any(len(v) > 5 for v in visual_texts) for s in subtitles])
    ref_keywords = ['see', 'shown', 'diagram', 'image', 'figure', 'visual']
    ref_count = sum(any(k in s['text'] for k in ref_keywords) for s in subtitles)
    match_count = sum(any(len(v.split()) > 5 for v in visual_texts) for s in subtitles if any(k in s['text'] for k in ref_keywords))
    annotation_accuracy = match_count / ref_count if ref_count else 0.6

    return {
        "visual_relevance_score": sim_score,
        "temporal_alignment_score": temporal_alignment,
        "conceptual_alignment_score": concept_overlap,
        "visual_annotation_accuracy": annotation_accuracy
    }

# ----------------------------
# ✅ MAIN VISUAL ANALYSIS FUNCTION
# ----------------------------

def analyze_video_visuals(video_path, subtitle_path=None, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scene_changes = 0
    prev_hist = None
    text_frame_count = 0
    face_presence_count = 0
    slide_durations = []
    object_distribution = {}
    emotion_distribution = {}

    processed_frames = range(0, total_frames, frame_interval)
    scene_start_frame = 0

    visual_texts = []
    slide_change_times = []

    for frame_num in processed_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correlation < 0.85:
                scene_changes += 1
                duration = (frame_num - scene_start_frame) / fps
                if duration > 0.5:
                    slide_durations.append(duration)
                    slide_change_times.append(frame_num / fps)
                scene_start_frame = frame_num
        prev_hist = hist

        results = MODEL(frame, verbose=False)[0]
        if results.boxes and results.boxes.cls is not None:
            for cls_id in results.boxes.cls.cpu().numpy():
                cls_name = MODEL.model.names[int(cls_id)]
                object_distribution[cls_name] = object_distribution.get(cls_name, 0) + 1

        if frame_num % int(fps) == 0:
            text = pytesseract.image_to_string(gray, config='--psm 6')
            if len(text.strip()) > 8 and len(re.findall(r'[a-zA-Z]', text)) >= 5:
                visual_texts.append(preprocess_text(text))
                text_frame_count += 1

        if frame_num % int(fps * 3) == 0:
            try:
                faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend=FACE_DETECTOR_BACKEND)
                if faces:
                    face_presence_count += 1
                    emotion_obj = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend=FACE_DETECTOR_BACKEND)
                    dominant_emotion = emotion_obj[0]['dominant_emotion']
                    emotion_distribution[dominant_emotion] = emotion_distribution.get(dominant_emotion, 0) + 1
            except Exception:
                continue

    cap.release()

    subtitle_scores = {}
    if subtitle_path:
        subtitles = parse_subtitles(subtitle_path)
        subtitle_scores = calculate_subtitle_alignment(subtitles, visual_texts, slide_change_times)

    return {
        "frames_with_text": text_frame_count,
        "total_scenes": scene_changes + 1,
        "face_presence_count": face_presence_count,
        "slide_durations": slide_durations,
        "object_distribution": object_distribution,
        "dominant_emotions": emotion_distribution,
        "total_frames": len(processed_frames),
        "fps": fps,  # ✅ Add this line
        **subtitle_scores
    }

# ----------------------------
# ✅ SCORING FUNCTION
# ----------------------------

def interpret_score(score, thresholds):
    for min_val, label in thresholds:
        if score >= min_val:
            return label
    return thresholds[-1][1]

def compute_conceptual_alignment(subtitle_texts, visual_texts):
    all_docs = [' '.join(subtitle_texts), ' '.join(visual_texts)]
    vec = TfidfVectorizer().fit_transform(all_docs)
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

def score_video_quality(metrics, total_frames, fps):
    slide_score = min(5.0, (metrics["frames_with_text"] / total_frames) * 5)
    slide_label = interpret_score(slide_score, [(4, "Excellent"), (3, "Good"), (2, "Average"), (0, "Poor")])

    video_duration_sec = total_frames / fps
    scene_density = metrics["total_scenes"] / (video_duration_sec / 60)
    scene_score = min(5.0, scene_density * 1.5)  # tuned for 3 scenes/min = 4.5
    object_variety_score = min(5.0, len(metrics["object_distribution"]) / 10 * 5)
    visual_score = (scene_score + object_variety_score) / 2
    visual_label = interpret_score(visual_score, [(4, "Excellent"), (3, "Good"), (2, "Average"), (0, "Poor")])

    instructor_ratio = metrics["face_presence_count"] / total_frames
    instructor_score = min(5.0, instructor_ratio * 20)
    instructor_label = interpret_score(instructor_score, [(4, "Excellent"), (3, "Good"), (2, "Average"), (0, "Poor")])

    positive_emotions = ['happy', 'neutral', 'calm']
    total_emotions = sum(metrics["dominant_emotions"].values())
    if total_emotions > 0:
        emotion_score = sum([metrics["dominant_emotions"].get(e, 0) for e in positive_emotions]) / total_emotions * 5
    else:
        emotion_score = 2.5
    emotion_score = min(5.0, emotion_score)
    emotion_label = interpret_score(emotion_score, [(4, "Positive"), (3, "Neutral"), (2, "Mixed"), (0, "Negative")])

    if metrics["slide_durations"]:
        avg_duration = sum(metrics["slide_durations"]) / len(metrics["slide_durations"])
        structure_score = min(5.0, max(0, 5.0 - avg_duration / 60))  # more lenient
    else:
        structure_score = 3.0
    structure_label = interpret_score(structure_score, [(4, "Structured"), (3, "Balanced"), (2, "Inconsistent"), (0, "Unstructured")])

    output = {
        "slide_clarity_score": {"score": round(slide_score, 2), "label": slide_label},
        "visual_engagement_score": {"score": round(visual_score, 2), "label": visual_label},
        "instructor_presence_score": {"score": round(instructor_score, 2), "label": instructor_label},
        "emotional_tone_score": {"score": round(emotion_score, 2), "label": emotion_label},
        "slide_structure_score": {"score": round(structure_score, 2), "label": structure_label}
    }

    for key in ["visual_relevance_score", "temporal_alignment_score", "conceptual_alignment_score", "visual_annotation_accuracy"]:
        if key in metrics:
            output[key] = round(metrics[key] * 5, 2)  # scale subtitle scores to 1–5

    return output

# Modified prompt functions to generate simpler, less detailed outputs
genai.configure(api_key=GOOGLE_API_KEY)  # only for API use

gemini_model = genai.GenerativeModel("gemini-2.0-flash-001")

def generate_score_feedback(score_name, score_value, raw_metrics=None):
    simplified_steps = {
        "slide_clarity_score": [
            f"The slide clarity score is {score_value} out of 5.",
            "This reflects how much readable text was detected on the slides.",
            "Low scores suggest slides may lack useful text or be hard to read."
        ],
        "visual_engagement_score": [
            f"The visual engagement score is {score_value} out of 5.",
            "It captures how visually varied and dynamic the video is.",
            "More scene changes and visual elements help keep learners engaged."
        ],
        "instructor_presence_score": [
            f"The instructor presence score is {score_value} out of 5.",
            "It shows how often the instructor appears on screen.",
            "Seeing the instructor often helps learners feel more connected."
        ],
        "emotional_tone_score": [
            f"The emotional tone score is {score_value} out of 5.",
            "This measures how positive or calm the expressions are.",
            "Friendly and calm faces can improve the learning atmosphere."
        ],
        "slide_structure_score": [
            f"The slide structure score is {score_value} out of 5.",
            "It reflects how consistently slides are paced and timed.",
            "Even pacing helps learners follow the content smoothly."
        ],
        "visual_consistency_scaled_1_to_5": [
            f"The visual consistency score is {score_value} out of 5.",
            "It shows how uniform the video visuals are.",
            "Consistent visuals help reduce distraction."
        ],
        "visual_relevance_score": [
            f"The visual relevance score is {score_value} out of 5.",
            "This shows how well visuals match the narration.",
            "Better visuals support understanding of the topic."
        ],
        "temporal_alignment_score": [
            f"The temporal alignment score is {score_value} out of 5.",
            "This measures how well slides and subtitles match over time.",
            "Good timing improves focus and clarity."
        ],
        "conceptual_alignment_score": [
            f"The conceptual alignment score is {score_value} out of 5.",
            "This reflects how well the visuals match the content.",
            "Stronger alignment means better explanations."
        ],
        "visual_annotation_accuracy": [
            f"The visual annotation accuracy score is {score_value} out of 5.",
            "It shows how clearly visuals are explained with text or voice.",
            "Clear references make it easier to follow diagrams and figures."
        ]
    }

    steps = simplified_steps.get(score_name, [f"Score {score_name} = {score_value}"])
    prompt = "\n".join([
        *steps,
        "",
        "Briefly explain how this affects the learner experience, and suggest 1–2 basic improvements."
    ])
    return prompt


def generate_instructor_feedback(module_name, visual_scores: dict, lo_metrics: dict):
    visual_score_feedback = "\n".join([
        f"- {k.replace('_', ' ').title()}: {v['score']} ({v['label']})"
        for k, v in visual_scores.items() if isinstance(v, dict)
    ])

    lo_metric_feedback = ""
    for metric in ['per_lo_scores_scaled_1_to_5', 'temporal_coverage_scaled_1_to_5', 'concept_density_scaled_1_to_5']:
        lo_metric_feedback += f"\n{metric.replace('_', ' ').title()}:\n"
        lo_metric_feedback += "\n".join([
            f"- {lo_text[:40]}...: {round(score, 2)}"
            for lo_text, score in lo_metrics.get(metric, {}).items()
        ]) + "\n"

    prompt = f"""
### Module: {module_name}

You're providing high-level improvement advice to the instructor. Be brief and constructive.

#### Visual Scores:
{visual_score_feedback}

#### Learning Objective Issues:
{lo_metric_feedback}

#### Instructions:
- Point out the 2–3 most important issues.
- Suggest one improvement for each.
- End with a one-sentence summary of what to fix first.
"""
    return prompt


def generate_learner_summary(module_name, visual_scores: dict, lo_metrics: dict):
    visual_summary_lines = []
    for k, v in visual_scores.items():
        label = v['label'] if isinstance(v, dict) else "N/A"
        score = v['score'] if isinstance(v, dict) else round(v, 2)
        visual_summary_lines.append(f"- {k.replace('_', ' ').title()}: {score} ({label})")

    lo_summary_lines = []
    for metric in ['per_lo_scores_scaled_1_to_5', 'temporal_coverage_scaled_1_to_5', 'concept_density_scaled_1_to_5']:
        lo_summary_lines.append(f"\n{metric.replace('_', ' ').title()}:")
        lo_summary_lines.extend([
            f"- {lo[:40]}...: {round(score, 2)}"
            for lo, score in lo_metrics.get(metric, {}).items()
        ])

    prompt = f"""
### Module: {module_name}

Write a short, friendly summary for learners browsing this module. Avoid any jargon.

#### Visuals:
{chr(10).join(visual_summary_lines)}

#### Learning Goals:
{chr(10).join(lo_summary_lines)}

#### Instructions:
- Write 2–3 short sentences.
- Keep it easy to read and informal.
- End with a simple verdict: Recommended, Partially Recommended, Not Recommended.
"""
    return prompt


def generate_learner_feedback_video_lo(score: float) -> str:
    return f"""
You're evaluating how well the video content supports the learning objectives overall.

Score (1–5): {round(score, 2)}

Write a short, friendly note for learners:
- What does this score mean for their learning experience?
- Should they feel confident, or are there some gaps?
- Use simple language and avoid jargon.
- Keep it within 2–3 sentences.
"""

def generate_instructor_feedback_video_lo(score: float) -> str:
    return f"""
You're reviewing how well this video aligns with the stated learning objectives overall.

Score (1–5): {round(score, 2)}

Write a short, constructive comment for the instructor:
- What does this score suggest about the video’s effectiveness?
- Suggest one possible way to improve alignment if needed.
- Keep the tone helpful and concise.
"""

import os
import json
from typing import List, Dict  # This is your full V3 code above

lo_analyzer = VideoLOAnalyzer()
# --------------------------------------------
# ✅ Metadata loader
# --------------------------------------------

def load_course_metadata(metadata_path: str) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_module_info(metadata: dict) -> List[Dict]:
    modules_info = []
    for key, value in metadata.items():
        if key.startswith("Module") and isinstance(value, dict):
            modules_info.append({
                "Module Name": value.get("Name", f"Unknown Module ({key})"),
                "Learning Objectives": value.get("Learning Objectives", [])
            })
    return modules_info

# --------------------------------------------
# ✅ File finder
# --------------------------------------------

def find_video_files(base_path: str) -> List[str]:
    video_paths = []
    for root, dirs, files in os.walk(base_path):
        if "index.mp4" in files and "subtitles-en.vtt" in files:
            video_paths.append(os.path.join(root, "index.mp4"))
    return video_paths

# --------------------------------------------
# ✅ Map videos to modules
# --------------------------------------------

def map_videos_to_modules(video_paths: List[str], module_infos: List[Dict]) -> List[Dict]:
    video_module_map = []
    for video_path in video_paths:
        parts = video_path.split('/')
        week_folder = next(part for part in parts if part.lower().startswith('week')).lower().strip()
        matched_module = next(
            (module for module in module_infos if module["Module Name"].lower().startswith(week_folder)),
            None
        )
        if matched_module:
            video_module_map.append({
                "video_path": video_path,
                "subtitle_path": os.path.join(os.path.dirname(video_path), "subtitles-en.vtt"),
                "module_name": matched_module["Module Name"],
                "learning_objectives": matched_module["Learning Objectives"]
            })
        else:
            print(f"Warning: No module match found for video: {video_path}")
    return video_module_map

def normalize_scores_to_1_5(visual_scores: dict) -> dict:
    scaled_output = {}
    for k, v in visual_scores.items():
        if isinstance(v, dict) and 'score' in v:
            # Scale from 0–10 to 1–5
            raw = v['score']
            scaled_output[k] = {
                'score': round(1 + 4 * (raw / 10), 2),
                'label': v['label']
            }
        elif isinstance(v, (int, float)):
            # Scale direct numerical fields (subtitle alignment metrics)
            scaled_output[k] = round(1 + 4 * (v / 10), 2)
        else:
            scaled_output[k] = v
    return scaled_output

# --------------------------------------------
# ✅ Analyze one video fully
# --------------------------------------------

def analyze_single_video(video_path, subtitle_path, learning_objectives, module_name="Unknown Module"):
    # Step 1: LO Analysis
    lo_analyzer = VideoLOAnalyzer()
    lo_metrics, _, _ = lo_analyzer.analyze_module(video_path, learning_objectives)

    # Step 2: Visual Analysis
    visual_metrics_raw = analyze_video_visuals(video_path, subtitle_path)
    visual_scores = score_video_quality(visual_metrics_raw, visual_metrics_raw["total_frames"], fps=visual_metrics_raw["fps"])
    visual_scores = normalize_scores_to_1_5(visual_scores)

    final_result = {}

    # Step 3: Visual metric feedback
    for score_name, score_obj in visual_scores.items():
        value = round(score_obj['score'] if isinstance(score_obj, dict) else score_obj, 2)
        learner_prompt = generate_score_feedback(score_name, value)
        instructor_prompt = generate_score_feedback(score_name, value)

        learner_summary = gemini_model.generate_content(learner_prompt).text
        instructor_feedback = gemini_model.generate_content(instructor_prompt).text

        final_result[score_name.replace("_", " ").title()] = {
            "Value": value,
            "Learner Summary": learner_summary,
            "Instructor Feedback": instructor_feedback
        }

    # Step 4: LO metric feedback (averaged)
    lo_aggregated = {}

    visual_relevance_scores = list(lo_metrics['per_lo_scores_scaled_1_to_5'].values())
    temporal_alignment_scores = list(lo_metrics['temporal_coverage_scaled_1_to_5'].values())
    conceptual_alignment_scores = list(lo_metrics['concept_density_scaled_1_to_5'].values())

    averages = {
        "Similarity Score": sum(visual_relevance_scores) / len(visual_relevance_scores) if visual_relevance_scores else -1,
        "Temporal Coverage": sum(temporal_alignment_scores) / len(temporal_alignment_scores) if temporal_alignment_scores else -1,
        "Concept Density": sum(conceptual_alignment_scores) / len(conceptual_alignment_scores) if conceptual_alignment_scores else -1,
    }

    lo_feedback = {}
    for metric_name, avg_score in averages.items():
        rounded_score = round(avg_score, 2)

        if metric_name == "Similarity Score":
            learner_prompt = generate_learner_feedback_video_lo(rounded_score)
            instructor_prompt = generate_instructor_feedback_video_lo(rounded_score)
        else:
            # For "Temporal Coverage" and "Concept Density", continue using general feedback
            learner_prompt = generate_score_feedback(metric_name, rounded_score)
            instructor_prompt = generate_score_feedback(metric_name, rounded_score)

        learner_fb = gemini_model.generate_content(learner_prompt).text
        instructor_fb = gemini_model.generate_content(instructor_prompt).text

        lo_feedback[metric_name] = {
            "Value": rounded_score,
            "Learner Summary": learner_fb,
            "Instructor Feedback": instructor_fb
        }

    final_result["LO Scores"] = lo_feedback

    # Step 5: Module-level summaries
    instructor_prompt = generate_instructor_feedback(module_name, visual_scores, lo_metrics)
    learner_prompt = generate_learner_summary(module_name, visual_scores, lo_metrics)

    instructor_summary = gemini_model.generate_content(instructor_prompt).text
    learner_summary = gemini_model.generate_content(learner_prompt).text

    final_result["Module Summary"] = {
        "Learner Summary": learner_summary,
        "Instructor Feedback": instructor_summary
    }

    lo_scores_output = {
        "Module Name": module_name,
        "LO Scores": lo_feedback
    }

    output_dir = os.path.join(os.path.dirname(video_path), "analysis_outputs")
    os.makedirs(output_dir, exist_ok=True)
    lo_scores_path = os.path.join(output_dir, "Video_LO_scores.json")

    with open(lo_scores_path, "w") as f:
        json.dump(lo_scores_output, f, indent=4)

    return final_result


# --------------------------------------------
# ✅ Average module-level results
# --------------------------------------------

def average_module_scores(video_results: List[Dict]) -> Dict:
    if not video_results:
        return {}

    def recursive_average(metrics_list):
        if isinstance(metrics_list[0], dict):
            keys = metrics_list[0].keys()
            return {k: recursive_average([d[k] for d in metrics_list]) for k in keys}
        elif isinstance(metrics_list[0], (int, float)):
            return sum(metrics_list) / len(metrics_list)
        return metrics_list[0]

    return recursive_average(video_results)

# --------------------------------------------
# ✅ Main Runner
# --------------------------------------------

def analyze_videos(mapped_data: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    module_to_results = {}

    for data in mapped_data:
        video_path = data["video_path"]
        subtitle_path = data["subtitle_path"]
        lo_list = data["learning_objectives"]
        module_name = data["module_name"]

        if not lo_list:
            print(f"Skipping {video_path} — No learning objectives found.")
            continue

        print(f"Analyzing {video_path} ...")
        module_name = os.path.basename(os.path.dirname(video_path))
        result = analyze_single_video(video_path, subtitle_path, lo_list, module_name)

        folder_name = os.path.basename(os.path.dirname(video_path))
        video_output_path = os.path.join(output_dir, f"{folder_name}.json")
        with open(video_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        module_to_results.setdefault(module_name, []).append(result)

    for module_name, results in module_to_results.items():
        averaged_result = average_module_scores(results)
        module_output_path = os.path.join(output_dir, f"{module_name}_summary.json")
        with open(module_output_path, "w", encoding="utf-8") as f:
            json.dump(averaged_result, f, indent=2)
        print(f"Saved summary for {module_name}")

# --------------------------------------------
# ✅ Full Pipeline Entry Point
# --------------------------------------------

def run_colab_pipeline(base_path):
    metadata_path = os.path.join(base_path, "metadata.json")
    output_dir = os.path.join(base_path, "Video LO Validation")

    metadata = load_course_metadata(metadata_path)
    module_infos = extract_module_info(metadata)
    video_paths = find_video_files(base_path)
    mapped_data = map_videos_to_modules(video_paths, module_infos)
    analyze_videos(mapped_data, output_dir)

# from colab_visual_lo_integrated import run_colab_pipeline

BASE_PATH = ''
run_colab_pipeline(BASE_PATH)