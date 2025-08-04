import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from typing import List, Dict, Tuple
import warnings
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
import json
import re
from googleapiclient.errors import HttpError
import http.client
from torch.utils.data import DataLoader, Dataset
import tempfile
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage import filters, feature
import io
from googleapiclient.http import MediaIoBaseDownload
from joblib import Parallel, delayed
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from google.api_core.exceptions import InternalServerError, ResourceExhausted
from sklearn.metrics.pairwise import cosine_similarity

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

warnings.filterwarnings('ignore')

try:
    from skimage import filters, feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Some features will be disabled.")
    SKIMAGE_AVAILABLE = False

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
                      if each_file["extension"] == ".mp4":
                        file_data["mp4_file"] = each_file["id"]
                  module_files[module_name].append(file_data)
    return module_files

class SOTAImageQualityAnalyzer:
    def __init__(self, device=None, batch_size=4):
        # Auto-detect device with fallback
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = 'cpu'
                print("Using CPU (GPU not available)")
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.models_loaded = False
        
        # Initialize models with error handling
        self._load_models()
        
    def _load_models(self):
        """Load all models with comprehensive error handling."""
        try:
            from transformers import (
                ViTImageProcessor, ViTForImageClassification,
                ConvNextImageProcessor, ConvNextForImageClassification,
                CLIPProcessor, CLIPModel
            )
            
            print(f"Loading models on {self.device}...")
            
            # ViT model with error handling
            try:
                self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
                self.vit_model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.vit_model = self.vit_model.half().to(self.device)
                else:
                    self.vit_model = self.vit_model.to(self.device)
                
                self.vit_model.eval()
                print("✓ ViT model loaded successfully")
                
            except Exception as e:
                print(f"Warning: Failed to load ViT model: {e}")
                self.vit_model = None
                self.vit_processor = None
            
            # ConvNeXt model with error handling
            try:
                self.convnext_processor = ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224')
                self.convnext_model = ConvNextForImageClassification.from_pretrained('facebook/convnext-base-224')
                
                # Move to device first, then convert to half precision
                self.convnext_model = self.convnext_model.to(self.device)
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.convnext_model = self.convnext_model.half()
                
                self.convnext_model.eval()
                print("✓ ConvNeXt model loaded successfully")
                
            except Exception as e:
                print(f"Warning: Failed to load ConvNeXt model: {e}")
                self.convnext_model = None
                self.convnext_processor = None
            
            # CLIP model with error handling
            try:
                self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                
                if self.device == 'cuda' and torch.cuda.is_available():
                    self.clip_model = self.clip_model.half().to(self.device)
                else:
                    self.clip_model = self.clip_model.to(self.device)
                
                self.clip_model.eval()
                
                # Pre-encode CLIP text features
                self._precompute_clip_features()
                print("✓ CLIP model loaded successfully")
                
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.clip_model = None
                self.clip_processor = None
                
            self.models_loaded = True
            
        except ImportError as e:
            print(f"Error: Transformers library not available: {e}")
            print("Please install: pip install transformers torch")
            self.models_loaded = False
        except Exception as e:
            print(f"Unexpected error loading models: {e}")
            self.models_loaded = False
    
    def _precompute_clip_features(self):
        """Pre-compute CLIP text features for educational context."""
        if self.clip_model is None or self.clip_processor is None:
            return
            
        educational_texts = [
            "a clear educational diagram",
            "a well-designed slide", 
            "educational content",
            "learning material",
            "instructional image"
        ]
        
        try:
            with torch.no_grad():
                text_inputs = self.clip_processor(
                    text=educational_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                self.clip_text_features = self.clip_model.get_text_features(**text_inputs)
                self.clip_text_features = self.clip_text_features / self.clip_text_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"Warning: Failed to precompute CLIP features: {e}")
            self.clip_text_features = None

    def load_images_parallel(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Load images in parallel with comprehensive error handling."""
        def load_single_image(path):
            try:
                if not os.path.exists(path):
                    return None, path, f"File not found: {path}"
                
                # Handle different image formats
                img = Image.open(path)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Validate image dimensions
                if img.size[0] < 32 or img.size[1] < 32:
                    return None, path, f"Image too small: {path} ({img.size})"
                
                # Resize if too large (memory optimization)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                return np.array(img), path, None
                
            except Exception as e:
                return None, path, f"Error loading {path}: {str(e)}"
        
        images = []
        valid_paths = []
        
        # Limit thread count to avoid memory issues
        max_workers = min(4, len(image_paths))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_single_image, path): path for path in image_paths}
            
            for future in as_completed(future_to_path):
                try:
                    img, path, error = future.result()
                    if img is not None:
                        images.append(img)
                        valid_paths.append(path)
                    elif error:
                        print(f"Warning: {error}")
                except Exception as e:
                    print(f"Error processing image: {e}")
        
        print(f"Successfully loaded {len(images)} images out of {len(image_paths)}")
        return images, valid_paths

    def preprocess_images_batch(self, images: List[np.ndarray]) -> Dict[str, any]:
        """Preprocess all images for all models with error handling."""
        preprocessed = {}
        
        try:
            pil_images = [Image.fromarray(img) for img in images]
            preprocessed['pil_images'] = pil_images
            
            # ViT preprocessing
            if self.vit_processor is not None:
                try:
                    vit_inputs = self.vit_processor(pil_images, return_tensors="pt")
                    # Move to device with error handling
                    vit_inputs = {k: v.to(self.device) for k, v in vit_inputs.items()}
                    preprocessed['vit_inputs'] = vit_inputs
                except Exception as e:
                    print(f"Warning: ViT preprocessing failed: {e}")
                    preprocessed['vit_inputs'] = None
            
            # ConvNeXt preprocessing
            if self.convnext_processor is not None:
                try:
                    convnext_inputs = self.convnext_processor(pil_images, return_tensors="pt")
                    convnext_tensors = convnext_inputs['pixel_values']
                    
                    # Ensure tensor dtype matches model precision
                    if self.device == 'cuda' and torch.cuda.is_available() and hasattr(self.convnext_model, 'dtype'):
                        if next(self.convnext_model.parameters()).dtype == torch.float16:
                            convnext_tensors = convnext_tensors.half()
                    
                    convnext_tensors = convnext_tensors.to(self.device)
                    preprocessed['convnext_tensors'] = convnext_tensors
                except Exception as e:
                    print(f"Warning: ConvNeXt preprocessing failed: {e}")
                    preprocessed['convnext_tensors'] = None
            
            # CLIP preprocessing
            if self.clip_processor is not None:
                try:
                    clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
                    clip_inputs = {k: v.to(self.device) for k, v in clip_inputs.items()}
                    preprocessed['clip_inputs'] = clip_inputs
                except Exception as e:
                    print(f"Warning: CLIP preprocessing failed: {e}")
                    preprocessed['clip_inputs'] = None
                    
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            
        return preprocessed

    def calculate_label_accuracy_sota_batch_with_intermediate(self, preprocessed: Dict) -> Tuple[float, List[float]]:
        """Calculate label accuracy and return intermediate per-frame confidences."""
        confidences = []
        num_images = len(preprocessed.get('pil_images', []))
        
        if num_images == 0:
            return 0.0, []
        
        try:
            with torch.no_grad():
                for i in range(0, num_images, self.batch_size):
                    end_idx = min(i + self.batch_size, num_images)
                    batch_confidences = []
                    
                    # ViT inference
                    if (self.vit_model is not None and preprocessed.get('vit_inputs') is not None):
                        try:
                            vit_batch = {k: v[i:end_idx] for k, v in preprocessed['vit_inputs'].items()}
                            vit_outputs = self.vit_model(**vit_batch)
                            vit_probs = F.softmax(vit_outputs.logits, dim=-1)
                            vit_confs = torch.max(vit_probs, dim=-1)[0]
                            batch_confidences.append(vit_confs)
                        except Exception as e:
                            print(f"Warning: ViT inference failed: {e}")
                    
                    # ConvNeXt inference
                    if (self.convnext_model is not None and preprocessed.get('convnext_tensors') is not None):
                        try:
                            convnext_batch = preprocessed['convnext_tensors'][i:end_idx]
                            model_dtype = next(self.convnext_model.parameters()).dtype
                            if convnext_batch.dtype != model_dtype:
                                convnext_batch = convnext_batch.to(dtype=model_dtype)
                            convnext_outputs = self.convnext_model(convnext_batch)
                            convnext_logits = convnext_outputs.logits
                            convnext_probs = F.softmax(convnext_logits, dim=-1)
                            convnext_confs = torch.max(convnext_probs, dim=-1)[0]
                            batch_confidences.append(convnext_confs)
                        except Exception as e:
                            print(f"Warning: ConvNeXt inference failed: {e}")
                    
                    # CLIP inference
                    if (self.clip_model is not None and preprocessed.get('clip_inputs') is not None and self.clip_text_features is not None):
                        try:
                            clip_batch = {k: v[i:end_idx] for k, v in preprocessed['clip_inputs'].items()}
                            clip_image_features = self.clip_model.get_image_features(**clip_batch)
                            clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
                            clip_similarities = torch.matmul(clip_image_features, self.clip_text_features.t())
                            clip_confs = torch.max(F.softmax(clip_similarities, dim=-1), dim=-1)[0]
                            batch_confidences.append(clip_confs)
                        except Exception as e:
                            print(f"Warning: CLIP inference failed: {e}")
                    
                    # Ensemble weighting
                    if batch_confidences:
                        if len(batch_confidences) == 3:
                            weights = torch.tensor([0.4, 0.4, 0.2], device=self.device)
                        elif len(batch_confidences) == 2:
                            weights = torch.tensor([0.5, 0.5], device=self.device)
                        else:
                            weights = torch.tensor([1.0], device=self.device)
                        batch_confs = torch.stack(batch_confidences, dim=1)
                        weighted_confs = torch.sum(batch_confs * weights[:len(batch_confidences)], dim=1)
                        confidences.extend(weighted_confs.cpu().numpy())
                    else:
                        confidences.extend([0.5] * (end_idx - i))
                        
        except Exception as e:
            print(f"Error in label accuracy calculation: {e}")
            return 0.5, []
        
        return float(np.mean(confidences)), confidences

    def calculate_extraneous_details_parallel_with_intermediate(self, images: List[np.ndarray]) -> Tuple[float, List[float]]:
        """Calculate extraneous details and return intermediate per-frame scores."""
        def process_single_image(img):
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                complexity_measures = []
                
                # Entropy calculation
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                complexity_measures.append(entropy / 8.0)
                
                # Edge density
                try:
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    complexity_measures.append(edge_density)
                except Exception:
                    complexity_measures.append(0.1)
                
                # Color variance
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    color_variance = np.mean([np.var(lab[:, :, i]) for i in range(3)])
                    complexity_measures.append(color_variance / 10000)
                except Exception:
                    complexity_measures.append(0.1)
                
                # Texture complexity
                if SKIMAGE_AVAILABLE:
                    try:
                        radius = 2
                        n_points = 8 * radius
                        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
                        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
                        texture_complexity = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
                        complexity_measures.append(texture_complexity / 10.0)
                    except Exception:
                        complexity_measures.append(0.1)
                else:
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    complexity_measures.append(min(laplacian_var / 1000, 1.0))
                
                # Frequency domain analysis
                try:
                    if gray.shape[0] * gray.shape[1] > 256 * 256:
                        gray_small = cv2.resize(gray, (256, 256))
                    else:
                        gray_small = gray
                    f_transform = np.fft.fft2(gray_small)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
                    frequency_complexity = np.std(magnitude_spectrum)
                    complexity_measures.append(frequency_complexity / 10.0)
                except Exception:
                    complexity_measures.append(0.1)
                
                # Weighted combination
                weights = [0.3, 0.2, 0.2, 0.15, 0.15]
                score = sum(w * c for w, c in zip(weights, complexity_measures[:len(weights)]))
                return min(score, 1.0)
                
            except Exception as e:
                print(f"Warning: Error processing image for extraneous details: {e}")
                return 0.1
        
        if not images:
            return 0.0, []
        
        try:
            max_workers = min(4, len(images))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                complexity_scores = list(executor.map(process_single_image, images))
            return float(np.mean(complexity_scores)), complexity_scores
        except Exception as e:
            print(f"Error in extraneous details calculation: {e}")
            return 0.1, []

    def calculate_visual_style_uniformity_sota_batch_with_intermediate(self, preprocessed: Dict) -> Tuple[float, List[float]]:
        """Calculate visual style uniformity and return intermediate similarity scores."""
        pil_images = preprocessed.get('pil_images', [])
        if len(pil_images) < 2:
            return 1.0, []
        
        try:
            num_images = len(pil_images)
            similarities = []
            clip_similarities_list = []
            color_similarities_list = []
            
            # CLIP-based similarity
            if (self.clip_model is not None and preprocessed.get('clip_inputs') is not None):
                try:
                    clip_features = []
                    with torch.no_grad():
                        for i in range(0, num_images, self.batch_size):
                            end_idx = min(i + self.batch_size, num_images)
                            clip_batch = {k: v[i:end_idx] for k, v in preprocessed['clip_inputs'].items()}
                            clip_embeddings = self.clip_model.get_image_features(**clip_batch)
                            clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)
                            clip_features.extend(clip_embeddings.cpu().numpy())
                    
                    if len(clip_features) > 1:
                        clip_sim_matrix = cosine_similarity(clip_features)
                        mask = ~np.eye(clip_sim_matrix.shape[0], dtype=bool)
                        clip_similarity = np.mean(clip_sim_matrix[mask])
                        similarities.append(('clip', clip_similarity, 0.6))
                        clip_similarities_list = clip_sim_matrix[mask].tolist()
                except Exception as e:
                    print(f"Warning: CLIP similarity calculation failed: {e}")
            
            # Color-based similarity
            try:
                def extract_color_features(img):
                    img_array = np.array(img)
                    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    hist_h = cv2.calcHist([img_hsv], [0], None, [30], [0, 180])
                    hist_s = cv2.calcHist([img_hsv], [1], None, [30], [0, 256])
                    hist_v = cv2.calcHist([img_hsv], [2], None, [30], [0, 256])
                    color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
                    return color_hist / (color_hist.sum() + 1e-8)
                
                color_features = [extract_color_features(img) for img in pil_images]
                color_sim_matrix = cosine_similarity(color_features)
                mask = ~np.eye(color_sim_matrix.shape[0], dtype=bool)
                color_similarity = np.mean(color_sim_matrix[mask])
                similarities.append(('color', color_similarity, 0.4))
                color_similarities_list = color_sim_matrix[mask].tolist()
            except Exception as e:
                print(f"Warning: Color similarity calculation failed: {e}")
                similarities.append(('color', 0.5, 0.4))
            
            # Weighted average
            if similarities:
                total_weight = sum(weight for _, _, weight in similarities)
                weighted_score = sum(sim * weight for _, sim, weight in similarities) / total_weight
                return max(0, min(1, weighted_score)), [clip_similarities_list, color_similarities_list]
            else:
                return 0.5, []
                
        except Exception as e:
            print(f"Error in visual style uniformity calculation: {e}")
            return 0.5, []

    def calculate_attention_retention_parallel_with_intermediate(self, images: List[np.ndarray]) -> Tuple[float, List[float]]:
        """Calculate attention retention and return intermediate per-frame scores."""
        def process_single_image(img):
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                attention_measures = []
                
                # Edge-based attention
                try:
                    edges = cv2.Canny(gray, 50, 150)
                    edge_variance = np.var(edges.astype(np.float32))
                    attention_measures.append(edge_variance / 10000)
                except Exception:
                    attention_measures.append(0.1)
                
                # Contrast-based attention
                try:
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    contrast_variance = np.var(laplacian)
                    attention_measures.append(contrast_variance / 10000)
                except Exception:
                    attention_measures.append(0.1)
                
                # Center bias simulation
                try:
                    h, w = gray.shape
                    y, x = np.ogrid[:h, :w]
                    center_bias = np.exp(-((x - w/2)**2 + (y - h/2)**2) / (2 * (min(h, w)/4)**2))
                    blurred = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
                    saliency_map = np.abs(gray.astype(np.float32) - blurred) * center_bias
                    saliency_variance = np.var(saliency_map)
                    attention_measures.append(saliency_variance / 1000)
                except Exception:
                    attention_measures.append(0.1)
                
                # Combine measures
                weights = [0.4, 0.3, 0.3]
                score = sum(w * m for w, m in zip(weights, attention_measures[:len(weights)]))
                return min(score, 1.0)
                
            except Exception as e:
                print(f"Warning: Error in attention calculation: {e}")
                return 0.1
        
        if not images:
            return 0.0, []
        
        try:
            max_workers = min(4, len(images))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                attention_scores = list(executor.map(process_single_image, images))
            
            if attention_scores and max(attention_scores) > 0:
                max_score = max(attention_scores)
                normalized_scores = [score / max_score for score in attention_scores]
                return float(np.mean(normalized_scores)), normalized_scores
            else:
                return 0.0, []
                
        except Exception as e:
            print(f"Error in attention retention calculation: {e}")
            return 0.1, []
        
    def map_metric_to_scale(self, value, metric_name):
        """
        Maps a metric value (0-1) to a float scale from 1.00 to 5.00.
        For 'extraneous_details', lower is better, so scale is inverted.

        Args:
            value (float): Metric value between 0 and 1.
            metric_name (str): One of ['label_accuracy', 'extraneous_details', 
                            'visual_style_uniformity', 'attention_retention_potential']

        Returns:
            float: Scaled rating from 1.00 (bad) to 5.00 (good)
        """
        # Invert for metrics where lower is better
        if metric_name == 'extraneous_details':
            value = 1 - value

        # Clamp between 0 and 1 to avoid runaway ratings
        value = max(0, min(1, float(value)))

        # Map to 1–5 scale (continuous)
        scale_value = 1 + (value * 4)

        return round(float(scale_value), 2)
    
    def analyze_images(self, image_paths: List[str]) -> dict:
        """
        Comprehensive SOTA quality analysis with intermediate outputs, mapped scores, and executed prompts.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Dictionary containing Value, User Feedback, and Instructor Feedback for each metric
        """
        if not self.models_loaded:
            print("Warning: Models not loaded properly. Results may be limited.")
        
        print("Loading images in parallel...")
        images, valid_paths = self.load_images_parallel(image_paths)
        
        if not images:
            print("No valid images found!")
            # Set default values for empty image case
            raw_metrics = {
                'label_accuracy': 0.0,
                'extraneous_details': 0.0,
                'visual_style_uniformity': 0.0,
                'attention_retention_potential': 0.0
            }
            intermediate_outputs = {
                'label_accuracy': [],
                'extraneous_details': [],
                'visual_style_uniformity': [[], []],
                'attention_retention_potential': []
            }
            mapped_scores = {
                'label_accuracy': 1,
                'extraneous_details': 1,
                'visual_style_uniformity': 1,
                'attention_retention_potential': 1
            }
        else:
            print(f"Analyzing {len(images)} images with SOTA methods...")
            
            # Preprocess all images once
            print("🔄 Preprocessing images...")
            preprocessed = self.preprocess_images_batch(images)
            
            # Calculate all metrics with intermediate outputs
            print("🔍 Calculating Label Accuracy...")
            label_acc, label_acc_intermediate = self.calculate_label_accuracy_sota_batch_with_intermediate(preprocessed)
            
            print("🎯 Calculating Extraneous Details...")
            extraneous_details, extraneous_intermediate = self.calculate_extraneous_details_parallel_with_intermediate(images)
            
            print("🎨 Calculating Visual Style Uniformity...")
            visual_style_uniformity, visual_style_intermediate = self.calculate_visual_style_uniformity_sota_batch_with_intermediate(preprocessed)
            
            print("👁️ Calculating Attention Retention...")
            attention_retention, attention_intermediate = self.calculate_attention_retention_parallel_with_intermediate(images)
            
            # Store raw metrics and intermediate outputs
            raw_metrics = {
                'label_accuracy': label_acc,
                'extraneous_details': extraneous_details,
                'visual_style_uniformity': visual_style_uniformity,
                'attention_retention_potential': attention_retention
            }
            
            intermediate_outputs = {
                'label_accuracy': label_acc_intermediate,
                'extraneous_details': extraneous_intermediate,
                'visual_style_uniformity': visual_style_intermediate,
                'attention_retention_potential': attention_intermediate
            }
            
            # Map to 1-5 scale
            mapped_scores = {
                'label_accuracy': self.map_metric_to_scale(label_acc, 'label_accuracy'),
                'extraneous_details': self.map_metric_to_scale(extraneous_details, 'extraneous_details'),
                'visual_style_uniformity': self.map_metric_to_scale(visual_style_uniformity, 'visual_style_uniformity'),
                'attention_retention_potential': self.map_metric_to_scale(attention_retention, 'attention_retention_potential')
            }

        print("🤖 Generating feedback prompts...")
        
        # Execute prompts for each metric
        label_accuracy_user_feedback = execute_prompt(prompt_label_accuracy(intermediate_outputs['label_accuracy'], mapped_scores['label_accuracy'])[0])
        label_accuracy_instructor_feedback = execute_prompt(prompt_label_accuracy(intermediate_outputs['label_accuracy'], mapped_scores['label_accuracy'])[1])
        
        extraneous_details_user_feedback = execute_prompt(prompt_extraneous_details(intermediate_outputs['extraneous_details'], mapped_scores['extraneous_details'])[0])
        extraneous_details_instructor_feedback = execute_prompt(prompt_extraneous_details(intermediate_outputs['extraneous_details'], mapped_scores['extraneous_details'])[1])
        
        visual_style_uniformity_user_feedback = execute_prompt(prompt_visual_style_uniformity(intermediate_outputs['visual_style_uniformity'], mapped_scores['visual_style_uniformity'])[0])
        visual_style_uniformity_instructor_feedback = execute_prompt(prompt_visual_style_uniformity(intermediate_outputs['visual_style_uniformity'], mapped_scores['visual_style_uniformity'])[1])
        
        attention_retention_user_feedback = execute_prompt(prompt_attention_retention(intermediate_outputs['attention_retention_potential'], mapped_scores['attention_retention_potential'])[0])
        attention_retention_instructor_feedback = execute_prompt(prompt_attention_retention(intermediate_outputs['attention_retention_potential'], mapped_scores['attention_retention_potential'])[1])

        print("✅ Analysis complete!")
        
        # Return results in the requested format
        results = {
            "Label Accuracy Score": {
                "Value": mapped_scores['label_accuracy'], 
                "User Feedback": label_accuracy_user_feedback, 
                "Instructor Feedback": label_accuracy_instructor_feedback
            },
            "Extraneous Details Score": {
                "Value": mapped_scores['extraneous_details'], 
                "User Feedback": extraneous_details_user_feedback, 
                "Instructor Feedback": extraneous_details_instructor_feedback
            },
            "Visual Style Uniformity Score": {
                "Value": mapped_scores['visual_style_uniformity'], 
                "User Feedback": visual_style_uniformity_user_feedback, 
                "Instructor Feedback": visual_style_uniformity_instructor_feedback
            },
            "Attention Retention Potential Score": {
                "Value": mapped_scores['attention_retention_potential'], 
                "User Feedback": attention_retention_user_feedback, 
                "Instructor Feedback": attention_retention_instructor_feedback
            }
        }
        
        return results

def execute_prompt(prompt):
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
    return response

def prompt_label_accuracy(intermediate_confidences, final_score):
    conf_min = min(intermediate_confidences) if intermediate_confidences else 0.0
    conf_max = max(intermediate_confidences) if intermediate_confidences else 1.0
    conf_avg = sum(intermediate_confidences) / len(intermediate_confidences) if intermediate_confidences else 0.0

    learner_prompt = f"""
    You are reviewing how clearly course visuals communicate educational content. The **Label Accuracy** metric quantifies how confidently advanced AI models recognize the frames as educational.

    **Methodology:**
    - Frame-by-frame, three state-of-the-art models (ViT, ConvNeXt, CLIP) estimate the probability your visual is recognized as educational.
    - Each frame receives a confidence score (range: 0.0 to 1.0).
    - The overall Label Accuracy is a weighted average of these scores, then mapped to a 1–5 scale (1 = poor clarity, 5 = excellent clarity).

    **Intermediate Score Range:**  
    Per-frame confidences ranged from **{conf_min}** to **{conf_max}** (average: **{conf_avg}**).

    **Final Metric Derivation:**  
    The average weighted confidence, reflecting all frames, is mapped directly to the 1–5 scale.

    **Directionality:**  
    Higher intermediate and final scores indicate strong visual clarity and alignment with educational themes; lower scores suggest frames are ambiguous or off-topic.

    **Input Data:**  
    - Per-frame Confidences: {intermediate_confidences}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Reflect on whether the visuals are recognized as clear, relevant educational content. If the average and range are close to 1.0, the material achieves industry-best clarity; scores substantially below 1.0 signal issues. Offer guidance on possible learning impact, referencing the scale (1 bad, 5 ideal).>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """

    instructor_prompt = f"""
    As a course auditor, you assess the clarity of educational visuals using **Label Accuracy**. This metric evaluates how reliably leading AI models recognize each frame as educational.

    **Methodology:**
    - ViT, ConvNeXt, and CLIP models assign per-frame confidence scores (0.0 to 1.0).
    - Weighted averaging produces a composite metric.
    - The final score is mapped to a 1–5 integer (1 = very poor, 5 = industry standard).

    **Intermediate Parameter Ranges:**  
    - Min: {conf_min}, Max: {conf_max}, Mean: {conf_avg}

    **Derivation Link:**  
    - The final score reflects the aggregated, weighted recognition quality across all frames, mapped to the standard 1–5 scale.
    
    **Directionality:**  
    - Scores near 5 mean the entire visual set consistently aligns with educational objectives; scores near 1 reveal content design or relevance failures.

    **Input Data:**  
    - Per-frame Confidences: {intermediate_confidences}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Diagnose which frames underperform, validate if the spread approaches the target of 1.0, and provide actionable clarity improvements. Reference if the mean or any frame’s value is far from the ideal (1.0).>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """
    return learner_prompt, instructor_prompt

def prompt_visual_style_uniformity(intermediate_similarities, final_score):
    clip_similarities = intermediate_similarities[0] if len(intermediate_similarities) > 0 else []
    color_similarities = intermediate_similarities[1] if len(intermediate_similarities) > 1 else []
    clip_min = min(clip_similarities) if clip_similarities else 0.0
    clip_max = max(clip_similarities) if clip_similarities else 1.0
    color_min = min(color_similarities) if color_similarities else 0.0
    color_max = max(color_similarities) if color_similarities else 1.0

    learner_prompt = f"""
    You are reviewing how consistent the course's visual style is. The **Visual Style Uniformity** metric measures similarity across frames, encouraging cohesiveness in colors, layouts, and design.

    **Methodology:**
    - For every pair of frames:
      - Calculate CLIP embedding similarity (semantic style; 0.0 to 1.0).
      - Measure color histogram similarity (color consistency; 0.0 to 1.0).
    - Both matrices are combined (weighted 60% CLIP, 40% color).
    - Higher similarity throughout implies strong uniformity.
    - The final score is mapped directly to the 1–5 scale (1 = inconsistent, 5 = cohesive).

    **Intermediate Parameter Ranges:**  
    - CLIP Similarity: {clip_min} to {clip_max}
    - Color Similarity: {color_min} to {color_max}

    **Score Derivation:**  
    - The closer all pairwise values are to 1.0, the higher the mapped score.

    **Directionality:**  
    - Score 5: Style is uniform across all visuals.
    - Score 1: Major inconsistencies across frames.

    **Input Data:**  
    - CLIP Similarities: {clip_similarities}
    - Color Similarities: {color_similarities}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Compare actual uniformity with the ideal (all pairs = 1.0). Indicate whether visuals appear seamlessly consistent or if there are jarring transitions, referencing the ranges. Relate the mapped score (1–5) to the implications for learning continuity.>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """

    instructor_prompt = f"""
    As a course design auditor, you validate the style consistency using **Visual Style Uniformity**.

    **Methodology:**
    - Cohesion is measured via pairwise CLIP and color histogram similarities (both 0.0–1.0).
    - Weighted average across all pairs.
    - Final uniformity mapped 1–5.

    **Parameter Ranges:**  
    - CLIP: Min {clip_min}, Max {clip_max}
    - Color: Min {color_min}, Max {color_max}

    **Final Metric Link:**  
    - Score rises as style similarity approaches 1.0 between all pairs. Mapped 1–5.

    **Directionality:**  
    - Score 5 signals enterprise-grade consistency throughout content. Score 1 reveals the need for major standardization.

    **Input Data:**  
    - CLIP Similarities: {clip_similarities}
    - Color Similarities: {color_similarities}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Diagnose which scenes are inconsistent, using similarity extremes as evidence. Compare to the ideal case; recommend targeted style alignment if variation exists.>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """
    return learner_prompt, instructor_prompt

def prompt_extraneous_details(intermediate_scores, final_score):
    ext_min = min(intermediate_scores) if intermediate_scores else 0.0
    ext_max = max(intermediate_scores) if intermediate_scores else 1.0
    ext_avg = sum(intermediate_scores) / len(intermediate_scores) if intermediate_scores else 0.0

    learner_prompt = f"""
    You are evaluating how visually clean course frames are. The **Extraneous Details** metric assesses the presence of unnecessary clutter, distraction, or complexity in each visual.

    **Methodology:**
    - Scores are calculated for each frame using:
      - Entropy (distribution of information, 0.0 to 1.0)
      - Edge density (proportion of detected edges, 0.0 to 1.0)
      - Color variance (color diversity, 0.0 to 1.0)
      - Texture complexity (local contrast, 0.0 to 1.0)
      - Frequency domain analysis (visual noise, 0.0 to 1.0)
    - The weighted sum is averaged across frames (lower is better).
    - The final score is inverted, then mapped to a 1–5 scale (1 = cluttered, 5 = clean).

    **Intermediate Parameter Range:**  
    Complexity scores range from **{ext_min}** to **{ext_max}**, with a mean of **{ext_avg}**.

    **Final Score Derivation:**  
    - Lower average complexity → Higher mapped score.
    - Higher average complexity → Lower mapped score.

    **Directionality:**  
    - Clean visuals (score 5) help maintain focus; cluttered ones (score 1) may hinder learning.

    **Input Data:**  
    - Per-frame Complexity Scores: {intermediate_scores}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Explain if the course visuals are as clean as the best-in-class standard (ideal mean & max = 0). Comment if any frames are excessive in clutter, referencing the observed range. Relate the final score on the 1–5 scale to learner effort and focus.>
   
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """

    instructor_prompt = f"""
    As a course design auditor, you assess visual simplicity using the **Extraneous Details** metric.

    **Methodology:**
    - Collective evaluation based on entropy, edge density, color variance, texture, and frequency complexity (each 0.0 to 1.0 per frame).
    - Scores are combined and averaged, then inverted for interpretability and mapped to 1–5 (1 = high clutter, 5 = minimal clutter).

    **Intermediate Range Check:**  
    - Min: {ext_min}, Max: {ext_max}, Mean: {ext_avg}

    **Metric Link:**  
    - The final score mirrors the cleanliness spread across all frames. Wider spread or higher mean suggests opportunity for simplification.

    **Directionality:**  
    - Approaching 5: visually simple, distraction-free. Near 1: excessive, possibly confusing detail.

    **Input Data:**  
    - Per-frame Complexity Scores: {intermediate_scores}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Explicitly identify outlier frames. Validate if the course approaches the ideal of 0 clutter. For low scores, recommend precise simplification targets using the parameter range as validation.>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """
    return learner_prompt, instructor_prompt

def prompt_attention_retention(intermediate_scores, final_score):
    att_min = min(intermediate_scores) if intermediate_scores else 0.0
    att_max = max(intermediate_scores) if intermediate_scores else 1.0
    att_avg = sum(intermediate_scores) / len(intermediate_scores) if intermediate_scores else 0.0

    learner_prompt = f"""
    You are reviewing how engaging and attention-guiding the visuals are. The **Attention Retention Potential** metric measures each frame’s ability to capture and hold viewer focus.

    **Methodology:**
    - Each frame is analyzed for:
      - Edge variance (detail sharpness, 0.0–1.0),
      - Contrast variance (visual punch, 0.0–1.0),
      - Center saliency (focal emphasis, 0.0–1.0).
    - Weighted average, batch-normalized, yields the per-frame score.
    - Final mapped score (1–5): Higher = more engaging.

    **Intermediate Parameter Range:**  
    Scores range from **{att_min}** to **{att_max}**, average **{att_avg}**.

    **Metric Link:**  
    The final potential is driven by the mean, directionality, and spread of these intermediate values.

    **Directionality:**  
    Higher mapped scores (closer to 5) mean frames have ideal center-of-interest and keep learners involved; lower (closer to 1) indicates risk of attention drift.

    **Input Data:**  
    - Per-frame Attention Scores: {intermediate_scores}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Validate to the user whether content reaches the ideal for focus and engagement (max mean = 1.0). If there are weak frames, describe their impact. Explicitly connect any shortfall in average or min scores to possible learning outcomes, using the 1–5 scale for context.>
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """

    instructor_prompt = f"""
    As an engagement auditor, you measure the attention retention strength using the **Attention Retention Potential** metric.

    **Methodology:**
    - Each frame’s edge variance, contrast variance, and center saliency are scored and averaged (all 0.0–1.0).
    - Final metric is mean of these, mapped to a 1–5 integer.

    **Intermediate Score Range:**  
    - Min: {att_min}, Max: {att_max}, Mean: {att_avg}

    **Final Score Link:**  
    - The mapped value summarizes how well the visuals can hold attention across the course.

    **Directionality:**  
    - 5: Highly engaging; 1: Flat or visually monotonous.

    **Input Data:**  
    - Per-frame Attention Scores: {intermediate_scores}
    - Final Score (1-5): {final_score}

    ---
    **Output Format (only this):**
    <Explicitly assess batch-level and frame-level engagement potential. Benchmark to the ideal case, and specify recommendations if engagement is not maximized.>
    
    Note: Please provide the response in a formal and user-friendly tone without unnecessary noise, suitable for direct communication with end users.
    """
    return learner_prompt, instructor_prompt

def get_scores(image_paths):
    """Example usage of the SOTA ImageQualityAnalyzer."""
    # Initialize SOTA analyzer
    analyzer = SOTAImageQualityAnalyzer(batch_size=4)
    # Perform SOTA analysis
    results = analyzer.analyze_images(image_paths)
    return results
  
def is_unique_fast(frame1, frame2, threshold=0.95):
    """Histogram comparison (10x faster than SSIM)"""
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0,256])
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return corr < threshold

def process_frame_batch(args):
    """Parallel frame processor"""
    cap, start, end, resize_factor, threshold = args
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    small_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    small_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    
    unique_frames = []
    prev_frame = None
    
    for _ in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (small_width, small_height))
        if prev_frame is None or is_unique_fast(prev_frame, small_frame, threshold):
            unique_frames.append(frame)
            prev_frame = small_frame
    return unique_frames

def extract_unique_frames_scores(video_path, threshold=0.95, 
                                    resize_factor=0.1, workers=os.cpu_count()-1, 
                                    batch_size=1000):
    """Ultra-fast parallel extraction"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Split work into batches
    batches = [(cap, i, min(i+batch_size, total_frames), 
                resize_factor, threshold) 
              for i in range(0, total_frames, batch_size)]
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(process_frame_batch, batches))
    
    # Save unique frames
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    for i, frame in enumerate(np.concatenate(results)):
        path = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(path, frame)
        image_paths.append(path)
    
    cap.release()
    scores = get_scores(image_paths)
    shutil.rmtree(temp_dir)
    return scores

def download_file_from_drive(file_id, service_account_json, dest_path):
    creds = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=creds)
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    fh.close()

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

def evaluate_images(video_path, service_account_file):
    temp_video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    download_file_from_drive(
        file_id=video_path,
        service_account_json=service_account_file,
        dest_path=temp_video_file.name
    )
    result_ = extract_unique_frames_scores(temp_video_file.name)
    temp_video_file.close()
    os.remove(temp_video_file.name)
    result_["File"] = get_parent_folder_name(video_path, service_account_file)
    return result_

class ImageAnalysisDataset(Dataset):
    def __init__(self, video_files):
        self.video_files = video_files

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        return self.video_files[idx]

def analyze_batch(batch, service_account_file):
    results = []
    for video_path in batch:
        try:
            result = evaluate_images(video_path, service_account_file)
            results.append(result)
        except Exception as e:
            print(f"Failed to process {video_path}: {str(e)}")
            results.append({"error": str(e)})
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

def perform_image_analysis(service_account_file, folder_id):
    result = []
    service = get_gdrive_service(service_account_file)
    # Try to fetch metadata.json
    metadata = fetch_metadata_json(service, folder_id)
    extensions = ['.mp4']
    all_files = find_files_recursively(service, folder_id, extensions)
    module_files = organize_files_by_module(metadata, all_files)
    print(module_files)
    image_summaries = []
    total_results = []
    for key, value in metadata.items():
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            module_results = {}
            module_results["Name"] = module_name
            module_results["Results"] = []
            print(f"Module --> {module_name}")
            video_files = [d["mp4_file"] for d in module_files[module_name] if "mp4_file" in d]
            dataset = ImageAnalysisDataset(video_files)
            loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x, num_workers=os.cpu_count()-1)
            for batch in loader:
                temp_ = analyze_batch(batch, service_account_file)
                module_results.extend(temp_)
                image_summaries.extend(temp_)
            total_results.append(module_results)
            break
    result["Module Results"] = total_results
    '''
    sums = defaultdict(int)
    counts = defaultdict(int)
    for d in image_summaries:
        for k, v in d.items():
            sums[k] += v
            counts[k] += 1
    overall_result = {}
    averages = {k: sums[k]/counts[k] for k in sums}
    for each in averages:
        overall_result[averages[each]] = {}
        value = averages[each]
        if isinstance(value, (np.floating, float)):
            value = float(value)
        averages[each] = str(value) + "(" + explain_metric_image(each, value) + ")"
        overall_result[averages[each]]["Value"] = int(value)
        overall_result[averages[each]]["Assessment"] = explain_metric_image(each, value)
    result["Overall Result"] = overall_result
    '''
    save_dict_to_json(result, "Instruction Material Agent - Image Quality Analysis.json")
    print("Image Quality Analysis Saved")

if __name__ == "__main__":
    service_account_file = ""
    folder_id = ""
    start_time = time.time()
    perform_image_analysis(service_account_file, folder_id)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Time taken: {elapsed_minutes:.2f} minutes")
