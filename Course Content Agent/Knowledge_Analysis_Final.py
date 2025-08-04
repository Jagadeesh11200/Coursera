import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import hdbscan
import torch
import google.generativeai as genai
from dataclasses import dataclass
import logging
from joblib import Parallel, delayed
import os
from collections import defaultdict, Counter
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from pypdf import PdfReader
import umap
import warnings
warnings.filterwarnings('ignore')
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
import http.client
import time
from io import BytesIO
import tempfile
import io
from googleapiclient.http import MediaIoBaseDownload
import base64
import subprocess
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import google.generativeai as genai

genai.configure(api_key = "")
model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleInput:
    course_info: str
    difficulty_level: str
    learning_objectives: List[str]
    transcript_file_paths: List[str]
    reading_contents: List[str]

@dataclass
class TopicCluster:
    cluster_id: int
    cluster_name: str
    keywords: List[str]
    importance_level: str
    recommended_content_ratio: float

@dataclass
class CCCSResult:
    final_score: float
    base_coverage: float
    confidence_bonus: float
    semantic_coherence_bonus: float
    coverage_bonus: float
    objective_alignment_bonus: float
    total_concepts: int
    covered_concepts: int


@dataclass
class RealWorldScenario:
    scenario_type: str
    content: str
    confidence_score: float
    file_source: str
    industry_domain: str
    detected_tools: List[str]

@dataclass
class TDBResult:
    tdb_score: float
    cluster_analysis: Dict[str, Dict[str, Any]]
    content_distribution: Dict[str, float]
    imbalance_penalties: Dict[str, float]
    recommendations: List[str]

@dataclass
class RWSSResult:
    rwss_score: float
    detected_scenarios: List[RealWorldScenario]
    scenario_distribution: Dict[str, int]
    industry_coverage: Dict[str, float]
    tool_diversity_score: float
    practical_application_score: float


@dataclass
class PADResult:
    final_score: float
    normalized_final_score: float
    transcript_score: float
    assignment_score: float
    reading_score: float
    objective_score: float
    weighted_combined_score: float
    difficulty_multiplier: float

def clean_json(json_):
    json_ = re.sub(r'`','',json_)
    json_ = re.sub('json','',json_)
    return json.loads(json_)

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
                      if each_file["extension"] == ".txt":
                        file_data["txt_file"] = each_file["id"]
                      if each_file["extension"] == ".pdf":
                        file_data["pdf_file"] = each_file["id"]
                      if each_file["extension"] == ".odt":
                        file_data["odt_file"] = each_file["id"]
                  module_files[module_name].append(file_data)
    return module_files

class DynamicModuleAnalyzer:
    def __init__(self, gemini_api_key: str, service_account_file):
        """
        Initialize analyzer with dynamic AI-driven pattern generation and optimized sentence transformers
        """
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.service_account_file = service_account_file

        # Load optimized sentence transformer model for speed and accuracy balance
        self.sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # Faster than all-mpnet-base-v2
        
        # Enable GPU if available for faster processing
        if torch.cuda.is_available():
            self.sentence_model = self.sentence_model.to('cuda')
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Pre-warm the model for faster first inference
        self.sentence_model.encode(["warmup text"], convert_to_tensor=True, device=self.device)

        # Optimized vectorizer for fallback scenarios
        self.vectorizer = TfidfVectorizer(
            max_features=200,  # Further reduced for speed
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.9
        )

        # Initialize dynamic patterns
        self.course_specific_patterns = {}
        self.industry_tools = {}
        self.domain_keywords = {}
        self.scenario_patterns = {}
        self.real_world_indicators = []
        self.practical_application_terms = []
        self.industry_contexts = []

    def read_transcript_file(self, file_id: str, max_retries=5) -> str:
        """Read transcript content from a file with error handling and content optimization"""
        service = get_gdrive_service(self.service_account_file)
        retries = 0
        while retries < max_retries:
            try:
                request = service.files().get_media(fileId=file_id)
                file_content = request.execute()
                
                if isinstance(file_content, bytes):
                    file_content = file_content.decode('utf-8')
                
                content = file_content.strip()
                # Clean and optimize content for sentence transformer processing
                content = re.sub(r'\n+', ' ', content)
                content = re.sub(r'\s+', ' ', content)
                
                # Limit content length for performance (first 6000 chars for better semantic coverage)
                return content[:6000] if len(content) > 6000 else content
                
            except (HttpError, http.client.IncompleteRead) as e:
                retries += 1
                if retries >= max_retries:
                    logger.error(f"Failed to read file {file_id} after {max_retries} attempts: {str(e)}")
                    return ""
                wait_time = 2 ** retries
                time.sleep(wait_time)

    def generate_course_specific_patterns(self, course_info: str, difficulty_level: str,
                                        learning_objectives: List[str]) -> Dict[str, Any]:
        """Generate AI-driven course-specific patterns optimized for semantic analysis"""
        prompt = f"""
        Generate semantic-rich concepts for educational content analysis using sentence transformers.
        
        Course Information: {course_info}
        Difficulty Level: {difficulty_level}
        Learning Objectives: {', '.join(learning_objectives)}

        Create a JSON response with semantically meaningful concepts that work well with sentence transformers:
        {{
            "core_concepts": ["concept1", "concept2", "concept3"],
            "industry_tools": ["tool1", "tool2", "tool3"],
            "domain_keywords": ["keyword1", "keyword2", "keyword3"],
            "practical_applications": ["application1", "application2"],
            "real_world_scenarios": ["scenario1", "scenario2"]
        }}

        Focus on:
        1. Complete phrases and concepts (not single words) for better semantic matching
        2. Domain-specific terminology that appears in educational transcripts
        3. Practical applications and real-world contexts
        4. Technical concepts with clear semantic meaning
        
        Generate 5-10 items per category for optimal performance.
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            patterns_data = self._parse_ai_response(response.text)

            # Store patterns optimized for sentence transformers
            self.course_specific_patterns = patterns_data.get("core_concepts", [])[:10]
            self.industry_tools = patterns_data.get("industry_tools", [])[:8]
            self.domain_keywords = patterns_data.get("domain_keywords", [])[:8]
            self.practical_application_terms = patterns_data.get("practical_applications", [])[:6]
            self.real_world_indicators = patterns_data.get("real_world_scenarios", [])[:6]

            logger.info(f"Generated {len(self.course_specific_patterns)} semantic concepts")
            return patterns_data

        except Exception as e:
            logger.error(f"Error generating patterns: {e}")
            return self._get_fallback_patterns()

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response with improved JSON extraction"""
        try:
            import json
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return self._extract_patterns_from_text(response_text)
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._extract_patterns_from_text(response_text)

    def _extract_patterns_from_text(self, text: str) -> Dict[str, Any]:
        """Extract semantic patterns from unstructured text"""
        patterns = {
            'core_concepts': [],
            'industry_tools': [],
            'domain_keywords': [],
            'practical_applications': [],
            'real_world_scenarios': []
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if any(term in line.lower() for term in ['core concept', 'technical concept']):
                current_section = 'core_concepts'
            elif any(term in line.lower() for term in ['industry tool', 'technology']):
                current_section = 'industry_tools'
            elif any(term in line.lower() for term in ['domain keyword', 'terminology']):
                current_section = 'domain_keywords'
            elif any(term in line.lower() for term in ['practical application', 'application']):
                current_section = 'practical_applications'
            elif any(term in line.lower() for term in ['real world', 'scenario']):
                current_section = 'real_world_scenarios'
            elif line.startswith(('-', '*', '•')) and current_section:
                concept = re.sub(r'^[-*•]\s*', '', line).strip()
                if concept and len(concept) > 3:
                    patterns[current_section].append(concept)
        
        return patterns

    def _get_fallback_patterns(self) -> Dict[str, Any]:
        """Generate fallback patterns when AI fails"""
        return {
            "core_concepts": ["programming concepts", "data structures", "algorithms", "software development"],
            "industry_tools": ["development tools", "frameworks", "libraries", "platforms"],
            "domain_keywords": ["implementation", "methodology", "best practices", "optimization"],
            "practical_applications": ["real world projects", "case studies", "hands on experience"],
            "real_world_scenarios": ["industry applications", "professional development", "practical examples"]
        }
    
    def calculate_semantic_similarity_optimized(self, concept: str, content: str) -> float:
        """Enhanced semantic similarity with moderate penalties and full content processing"""
        if not content.strip():
            return 0.0
        
        # Method 1: Balanced keyword pre-filtering
        concept_lower = concept.lower()
        content_lower = content.lower()
        
        # Moderate keyword density check
        concept_words = concept_lower.split()
        keyword_matches = sum(1 for word in concept_words if word in content_lower)
        exact_matches = sum(content_lower.count(word) for word in concept_words)
        
        # Enhanced keyword scoring
        keyword_density = keyword_matches / len(concept_words) if concept_words else 0
        keyword_frequency = min(exact_matches / len(concept_words), 3.0) if concept_words else 0
        keyword_score = min((keyword_density * 0.6 + keyword_frequency * 0.4) / 1.5, 1.0)
        
        # Reasonable early exit threshold
        if keyword_score > 0.85 and len(concept_words) > 1:
            return min(keyword_score * 0.95 + 0.05, 1.0)
        
        # Method 2: Full content processing without chunking
        # Limit content length for performance while maintaining semantic richness
        if len(content) > 6000:
            # Take strategic samples: beginning, middle, end for comprehensive coverage
            third = len(content) // 3
            sampled_content = content[:2000] + " " + content[third:third+2000] + " " + content[-2000:]
        else:
            sampled_content = content
        
        # Method 3: Sentence transformer processing with moderate normalization
        try:
            # Prepare texts for encoding - no chunking, full semantic context
            texts_to_encode = [f"Educational concept: {concept}", sampled_content]
            
            # Batch encoding for efficiency
            embeddings = self.sentence_model.encode(
                texts_to_encode,
                batch_size=2,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            concept_embedding = embeddings[0:1]
            content_embedding = embeddings[1:2]
            
            # Calculate semantic similarity
            semantic_similarity = torch.mm(concept_embedding, content_embedding.T).item()
            
            # Moderate normalization - less aggressive than before
            # Transform from typical ST range [0.3, 1.0] to [0.1, 1.0] for better distribution
            normalized_semantic = max((semantic_similarity - 0.3) / 0.7, 0.0)
            
            # Method 4: Balanced score combination with light penalties
            # Apply only moderate quality penalty
            quality_penalty = self._calculate_moderate_quality_penalty(content, concept)
            
            # Intelligent score combination
            if normalized_semantic > 0.6:
                # Strong semantic match - favor semantic score
                final_score = 0.3 * keyword_score + 0.7 * normalized_semantic
            elif normalized_semantic > 0.3:
                # Moderate semantic match - balanced approach
                final_score = 0.5 * keyword_score + 0.5 * normalized_semantic
            else:
                # Weak semantic match - favor keyword score
                final_score = 0.7 * keyword_score + 0.3 * normalized_semantic
            
            # Apply moderate penalty
            penalized_score = final_score - quality_penalty
            
            # Reasonable final scoring with light boost
            return max(min(penalized_score * 1.1 + 0.05, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            # Reasonable fallback
            return min(keyword_score * 0.9 + 0.1, 0.8)

    def _calculate_moderate_quality_penalty(self, content: str, concept: str) -> float:
        """Calculate moderate penalty based on content quality"""
        penalty = 0.0
        
        words = content.lower().split()
        if len(words) == 0:
            return 0.2
        
        # Light penalty for very repetitive content only
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        if repetition_ratio < 0.2:  # Only very repetitive content
            penalty += 0.1  # Reduced from 0.2
        
        # Light penalty for extreme keyword stuffing only
        concept_words = concept.lower().split()
        total_concept_occurrences = sum(content.lower().count(word) for word in concept_words)
        concept_density = total_concept_occurrences / len(words)
        if concept_density > 0.15:  # Only extreme cases
            penalty += 0.08  # Reduced from 0.15
        
        return min(penalty, 0.15)  # Cap total penalty at 15%

    def batch_analyze_concepts_with_transformers_1(self, concepts: List[str], 
                                              transcript_contents: Dict[str, str]) -> Dict:
        """Balanced batch analysis with moderate penalties"""
        coverage_results = {}
        
        # Process content with reasonable limits
        processed_contents = {}
        for file_path, content in transcript_contents.items():
            if content.strip():
                # Keep more content for better semantic analysis
                processed_contents[file_path] = content
        
        # Analyze each concept with balanced scoring
        for concept in concepts:
            concept_scores = {}
            file_scores = []
            
            # Process all files for this concept
            for file_path, content in processed_contents.items():
                score = self.calculate_semantic_similarity_optimized(concept, content)
                concept_scores[file_path] = score
                file_scores.append(score)
            
            # Balanced aggregation without excessive penalties
            if file_scores:
                max_score = max(file_scores)
                avg_score = np.mean(file_scores)
                median_score = np.median(file_scores)
                std_score = np.std(file_scores)
                
                # Light consistency penalty only for very inconsistent results
                consistency_penalty = min(std_score * 0.2, 0.08) if std_score > 0.4 else 0.0
                
                # Balanced aggregation based on evidence strength
                if max_score > 0.7:
                    # Strong evidence - emphasize max score
                    overall_score = 0.6 * max_score + 0.3 * avg_score + 0.1 * median_score
                elif max_score > 0.4:
                    # Moderate evidence - balanced approach
                    overall_score = 0.4 * max_score + 0.4 * avg_score + 0.2 * median_score
                else:
                    # Weak evidence - conservative but not overly penalized
                    overall_score = 0.3 * max_score + 0.7 * avg_score
                
                # Apply light consistency penalty
                overall_score = max(overall_score - consistency_penalty, 0.0)
            else:
                overall_score = 0.0
            
            # Reasonable coverage threshold
            threshold = 0.3 if len(file_scores) > 2 else 0.25  # Reduced from 0.4/0.35
            
            coverage_results[concept] = {
                'overall_score': overall_score,
                'file_scores': concept_scores,
                'covered': overall_score >= threshold,
                'confidence': self._calculate_balanced_confidence_level(file_scores, overall_score)
            }
        
        return coverage_results

    def _download_pdf_to_temp_gdrive(self, service, file_id: str) -> dict:
        """Download PDF from Google Drive using service to temporary storage and prepare for Gemini"""
        try:
            # Request file media from Google Drive
            request = service.files().get_media(fileId=file_id)
            
            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                # Use BytesIO for in-memory download
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
                
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Download {int(status.progress() * 100)}%")
                
                # Write downloaded content to temporary file
                fh.seek(0)
                temp_file.write(fh.getvalue())
                temp_file_path = temp_file.name
            
            # Prepare PDF data for Gemini
            with open(temp_file_path, "rb") as pdf_file:
                pdf_data = {
                    "mime_type": "application/pdf",
                    "data": pdf_file.read(),
                }
            
            # Clean up temporary file
            import os
            os.unlink(temp_file_path)
            
            return pdf_data
            
        except Exception as e:
            print(f"Failed to download PDF {file_id} from Google Drive: {e}")
            return None

    def _analyze_concept_with_gemini(self, concept: str, pdf_data: dict) -> float:
        """Analyze concept coverage in PDF using hybrid approach: Gemini AI + ML techniques"""
        
        # First, extract text content from PDF using Gemini
        text_content = self._extract_text_with_gemini(pdf_data)
        
        if not text_content:
            return 0.0
        
        # Apply multiple ML-based scoring techniques
        scores = []
        
        # 1. TF-IDF based semantic similarity
        tfidf_score = self._calculate_tfidf_similarity(concept, text_content)
        scores.append(tfidf_score)
        
        # 2. Sentence transformer similarity
        sentence_score = self._calculate_sentence_similarity(concept, text_content)
        scores.append(sentence_score)
        
        # 3. Keyword frequency analysis
        keyword_score = self._calculate_keyword_frequency_score(concept, text_content)
        scores.append(keyword_score)
        
        # 4. Context window analysis
        context_score = self._calculate_context_coverage(concept, text_content)
        scores.append(context_score)
        
        # 5. Optional: Gemini-based analysis as additional signal (with fallback)
        gemini_score = self._get_gemini_coverage_score(concept, pdf_data)
        if gemini_score > 0:
            scores.append(gemini_score)
        
        # Weighted ensemble of all scores
        final_score = self._ensemble_scores(scores, text_content, concept)
        
        return max(0.0, min(1.0, final_score))

    def _extract_text_with_gemini(self, pdf_data: dict) -> str:
        """Extract text content from PDF using Gemini for ML processing"""
        try:
            prompt = """
            Extract all text content from this PDF document. 
            Return only the plain text content without any formatting or analysis.
            Preserve paragraph structure with line breaks.
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            message_parts = [prompt, pdf_data]
            response = model.generate_content(message_parts)
            
            return response.text.strip() if response.text else ""
        except Exception as e:
            print(f"Failed to extract text with Gemini: {e}")
            return ""

    def _calculate_tfidf_similarity(self, concept: str, text_content: str) -> float:
        """Calculate TF-IDF based similarity score"""
        try:
            # Split text into sentences for better granularity
            sentences = re.split(r'[.!?]+', text_content)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=1000,
                lowercase=True
            )
            
            # Add concept as reference document
            documents = [concept] + sentences
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity between concept and each sentence
            concept_vector = tfidf_matrix[0:1]
            sentence_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(concept_vector, sentence_vectors)[0]
            
            # Return weighted score based on top similarities
            top_similarities = sorted(similarities, reverse=True)[:5]
            if top_similarities:
                # Weight top matches more heavily
                weights = [0.4, 0.25, 0.2, 0.1, 0.05]
                weighted_score = sum(sim * weight for sim, weight in 
                                    zip(top_similarities, weights[:len(top_similarities)]))
                return min(weighted_score * 1.5, 1.0)  # Scale up slightly
            
            return 0.0
        except Exception as e:
            print(f"TF-IDF calculation error: {e}")
            return 0.0

    def _calculate_sentence_similarity(self, concept: str, text_content: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            # Load pre-trained sentence transformer model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
            # Split text into chunks for processing
            chunks = self._split_text_into_chunks(text_content, max_length=500)
            
            if not chunks:
                return 0.0
            
            # Encode concept and text chunks
            concept_embedding = model.encode([concept])
            chunk_embeddings = model.encode(chunks)
            
            # Calculate similarities
            similarities = cosine_similarity(concept_embedding, chunk_embeddings)[0]
            
            # Return score based on best matches
            max_similarity = max(similarities)
            avg_top_3 = np.mean(sorted(similarities, reverse=True)[:3])
            
            # Combine max and average for balanced scoring
            return 0.6 * max_similarity + 0.4 * avg_top_3
            
        except Exception as e:
            print(f"Sentence similarity calculation error: {e}")
            return 0.0

    def _calculate_keyword_frequency_score(self, concept: str, text_content: str) -> float:
        """Calculate score based on keyword frequency and density"""
        try:
            # Extract keywords from concept
            concept_words = set(re.findall(r'\b\w+\b', concept.lower()))
            concept_words = {word for word in concept_words if len(word) > 2}
            
            # Clean and tokenize text
            text_words = re.findall(r'\b\w+\b', text_content.lower())
            total_words = len(text_words)
            
            if total_words == 0 or not concept_words:
                return 0.0
            
            # Count keyword occurrences
            keyword_counts = {}
            for word in concept_words:
                keyword_counts[word] = text_words.count(word)
            
            # Calculate frequency-based score
            total_matches = sum(keyword_counts.values())
            frequency_score = min(total_matches / (total_words * 0.01), 1.0)  # Normalize
            
            # Calculate coverage score (how many concept words are present)
            coverage_score = len([w for w in concept_words if keyword_counts[w] > 0]) / len(concept_words)
            
            # Combine frequency and coverage
            return 0.7 * frequency_score + 0.3 * coverage_score
            
        except Exception as e:
            print(f"Keyword frequency calculation error: {e}")
            return 0.0

    def _calculate_context_coverage(self, concept: str, text_content: str) -> float:
        """Analyze concept coverage in context windows"""
        try:
            concept_words = set(re.findall(r'\b\w+\b', concept.lower()))
            sentences = re.split(r'[.!?]+', text_content)
            
            context_scores = []
            
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                
                # Check if any concept words appear in this sentence
                if any(word in sentence_lower for word in concept_words):
                    # Analyze surrounding context (±2 sentences)
                    start_idx = max(0, i-2)
                    end_idx = min(len(sentences), i+3)
                    context_window = ' '.join(sentences[start_idx:end_idx])
                    
                    # Score based on concept word density in context
                    context_words = re.findall(r'\b\w+\b', context_window.lower())
                    concept_density = sum(1 for word in context_words if word in concept_words)
                    
                    if len(context_words) > 0:
                        density_score = concept_density / len(context_words)
                        context_scores.append(min(density_score * 10, 1.0))  # Scale up
            
            return np.mean(context_scores) if context_scores else 0.0
            
        except Exception as e:
            print(f"Context coverage calculation error: {e}")
            return 0.0

    def _get_gemini_coverage_score(self, concept: str, pdf_data: dict) -> float:
        """Fallback Gemini analysis with simplified prompt"""
        try:
            prompt = f"""
            Rate how well this document covers the concept "{concept}" on a scale of 0.0 to 1.0.
            Consider: direct mentions, explanations, examples, and related content.
            Respond with only a number between 0.0 and 1.0.
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            message_parts = [prompt, pdf_data]
            response = model.generate_content(message_parts)
            
            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', response.text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            
            return 0.0
        except Exception as e:
            print(f"Gemini scoring error: {e}")
            return 0.0

    def _ensemble_scores(self, scores: list, text_content: str, concept: str) -> float:
        """Combine multiple scores using weighted ensemble"""
        if not scores:
            return 0.0
        
        # Remove zero scores for better averaging
        non_zero_scores = [s for s in scores if s > 0]
        
        if not non_zero_scores:
            return 0.0
        
        # Adaptive weighting based on text length and concept complexity
        text_length = len(text_content.split())
        concept_complexity = len(concept.split())
        
        if text_length > 1000 and concept_complexity > 2:
            # For long texts and complex concepts, emphasize semantic similarity
            weights = [0.35, 0.3, 0.15, 0.15, 0.05]  # TF-IDF, Sentence, Keyword, Context, Gemini
        else:
            # For shorter texts, balance all approaches
            weights = [0.25, 0.25, 0.25, 0.15, 0.1]
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in 
                        zip(scores, weights[:len(scores)]))
        
        # Apply confidence boost for consistent high scores
        if len(non_zero_scores) >= 3 and np.mean(non_zero_scores) > 0.6:
            weighted_sum *= 1.1  # 10% boost for consistent high scores
        
        return min(weighted_sum, 1.0)

    def _split_text_into_chunks(self, text: str, max_length: int = 500) -> list:
        """Split text into manageable chunks for processing"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _calculate_balanced_confidence_level(self, file_scores: List[float], overall_score: float) -> str:
        """Calculate confidence level based on score consistency and magnitude"""
        if not file_scores:
            return "low"
        
        avg_score = np.mean(file_scores)
        std_score = np.std(file_scores)
        
        # High confidence: high average score with low variance
        if avg_score > 0.7 and std_score < 0.15:
            return "high"
        # Medium confidence: moderate scores or high scores with moderate variance
        elif avg_score > 0.5 and std_score < 0.25:
            return "medium"
        # Low confidence: low scores or high variance
        else:
            return "low"

    def _calculate_balanced_confidence_level(self, file_scores: List[float], overall_score: float) -> str:
        """Balanced confidence level calculation"""
        if not file_scores:
            return 'low'
        
        max_score = max(file_scores)
        avg_score = np.mean(file_scores)
        score_variance = np.var(file_scores) if len(file_scores) > 1 else 0
        
        # Reasonable confidence thresholds
        if max_score > 0.7 and overall_score > 0.6 and score_variance < 0.08:
            return 'high'
        elif max_score > 0.5 and overall_score > 0.4 and score_variance < 0.15:
            return 'medium'
        else:
            return 'low'

    def calculate_cccs_full(self, module_input: ModuleInput) -> float:
        """
        Balanced CCCS calculation with moderate penalties and reasonable scoring
        """
        logger.info("Starting balanced CCCS calculation...")
        
        # Step: Balanced concept selection
        all_concepts = []
        
        # Reasonable concept quantities
        all_concepts.extend(self.course_specific_patterns)
        all_concepts.extend(self.practical_application_terms)
        all_concepts.extend(self.industry_tools)
        all_concepts.extend(self.domain_keywords)
        
        # Objective-derived concepts
        for obj in module_input.learning_objectives:
            phrases = re.findall(r'\b[A-Za-z][A-Za-z\s]{4,20}\b', obj)
            all_concepts.extend(phrases)  # Top 2 phrases per objective
        
        # Clean and limit concepts reasonably
        unique_concepts = []
        seen = set()
        for concept in all_concepts:
            if concept and len(concept.strip()) > 3:  # Reasonable minimum length
                clean_concept = concept.strip()
                if clean_concept.lower() not in seen:
                    unique_concepts.append(clean_concept)
                    seen.add(clean_concept.lower())
        
        # Reasonable limit for quality analysis
        unique_concepts = unique_concepts[:18]
        
        logger.info(f"Analyzing {len(unique_concepts)} concepts with balanced approach")
        
        # Step: Process transcripts
        transcript_contents = {}
        for path in module_input.transcript_file_paths:
            content = self.read_transcript_file(path)
            if content.strip():
                transcript_contents[path] = content
        
        reading_contents = {k: v for d in module_input.reading_contents for k, v in d.items()}
        transcript_contents = {**transcript_contents, **reading_contents}
        
        # Step: Balanced analysis
        coverage_results = self.batch_analyze_concepts_with_transformers_1(
            unique_concepts, transcript_contents
        )

        # Step: Balanced CCCS calculation
        covered_concepts = [concept for concept, result in coverage_results.items() 
                          if result['covered']]
        
        # Base coverage with reasonable calculation
        base_coverage = len(covered_concepts) / len(unique_concepts) if unique_concepts else 0.0
        
        # Moderate bonuses
        confidence_scores = {concept: result['overall_score'] 
                          for concept, result in coverage_results.items()}
        
        # Reasonable confidence bonuses
        high_confidence_count = sum(1 for result in coverage_results.values() 
                                  if result['confidence'] == 'high')
        medium_confidence_count = sum(1 for result in coverage_results.values() 
                                    if result['confidence'] == 'medium')
        
        confidence_bonus = (high_confidence_count * 0.04 + medium_confidence_count * 0.02)
        
        # Moderate bonuses
        semantic_coherence_bonus = self._calculate_semantic_coherence_bonus(covered_concepts) * 0.8
        coverage_bonus = min(len(covered_concepts) * 0.02, 0.12)
        objective_alignment_bonus = self._calculate_enhanced_objective_alignment(
            module_input.learning_objectives, covered_concepts, confidence_scores
        ) * 0.8
        
        # Balanced final score with reasonable baseline
        final_score = min(
            base_coverage * 1.1 +  # Light boost
            confidence_bonus + 
            semantic_coherence_bonus + 
            coverage_bonus + 
            objective_alignment_bonus + 
            0.08,  # Reasonable baseline
            0.95   # Realistic maximum
        )
        
        logger.info(f"Balanced CCCS Score: {final_score:.3f} | Covered: {len(covered_concepts)}/{len(unique_concepts)}")
        
        return CCCSResult(
            final_score=final_score,
            base_coverage=base_coverage,
            confidence_bonus=confidence_bonus,
            semantic_coherence_bonus=semantic_coherence_bonus,
            coverage_bonus=coverage_bonus,
            objective_alignment_bonus=objective_alignment_bonus,
            total_concepts=len(unique_concepts),
            covered_concepts=len(covered_concepts)
        )

    def _calculate_semantic_coherence_bonus(self, covered_concepts: List[str]) -> float:
        """Calculate semantic coherence bonus using sentence transformers"""
        if len(covered_concepts) < 2:
            return 0.0
        
        try:
            # Encode covered concepts for coherence analysis
            concept_embeddings = self.sentence_model.encode(
                covered_concepts,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
            
            # Calculate pairwise semantic similarities
            similarity_matrix = torch.mm(concept_embeddings, concept_embeddings.T)
            
            # Get average similarity excluding diagonal
            mask = ~torch.eye(len(covered_concepts), dtype=bool, device=self.device)
            avg_coherence = similarity_matrix[mask].mean().item()
            
            # Bonus based on semantic coherence (well-related concepts)
            coherence_bonus = min(avg_coherence * 0.06, 0.04)  # Up to 4% bonus
            
            return coherence_bonus
            
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0

    def _calculate_enhanced_objective_alignment(self, learning_objectives: List[str], 
                                              covered_concepts: List[str],
                                              confidence_scores: Dict[str, float]) -> float:
        """Enhanced objective alignment using semantic similarity"""
        if not learning_objectives or not covered_concepts:
            return 0.0
        
        try:
            # Encode objectives and concepts for semantic alignment
            all_texts = learning_objectives + covered_concepts
            
            embeddings = self.sentence_model.encode(
                all_texts,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
            
            obj_embeddings = embeddings[:len(learning_objectives)]
            concept_embeddings = embeddings[len(learning_objectives):]
            
            # Calculate cross-similarities between objectives and concepts
            alignment_similarities = torch.mm(obj_embeddings, concept_embeddings.T)
            
            # Get best alignment for each objective
            best_alignments = torch.max(alignment_similarities, dim=1)[0]
            avg_alignment = torch.mean(best_alignments).item()
            
            # Weight by concept confidence scores
            concept_confidences = [confidence_scores.get(concept, 0.0) for concept in covered_concepts]
            weighted_alignment = avg_alignment * (np.mean(concept_confidences) if concept_confidences else 0.0)
            
            return min(weighted_alignment * 0.1, 0.06)  # Up to 6% bonus
            
        except Exception as e:
            logger.error(f"Error calculating enhanced objective alignment: {e}")
            return 0.0

    # TDB Implementation (HDBSCAN clustering)
    def download_pdf_from_gdrive(self, service, file_id: str) -> str:
        """Download PDF from Google Drive to temporary storage"""
        try:
            request = service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            file_buffer.seek(0)
            temp_file.write(file_buffer.read())
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading PDF {file_id}: {e}")
            return None

    def process_pdf_with_gemini(self, pdf_path: str) -> str:
        """Process PDF with Gemini while preserving format"""
        try:
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = {
                    "mime_type": "application/pdf",
                    "data": pdf_file.read(),
                }
            
            prompt = """
            Analyze this PDF document and extract the key topics, concepts, and learning content.
            Focus on identifying:
            1. Main topics and subtopics
            2. Key concepts and terminology
            3. Learning objectives or goals mentioned
            4. Content structure and organization
            
            Provide a comprehensive summary of the educational content without losing important details.
            """
            
            model = genai.GenerativeModel("models/gemini-1.5-flash", 
                                        generation_config={"temperature": 0.0})
            message_parts = [prompt, pdf_data]
            response = model.generate_content(message_parts)
            
            return response.text.strip() if response.text else ""
        except Exception as e:
            logger.error(f"Error processing PDF with Gemini: {e}")
            return ""

    def extract_content_embeddings_with_readings(self, file_paths: List[str], 
                                            reading_contents) -> List[np.ndarray]:
        """Extract embeddings for transcripts and reading materials"""
        logger.info("Extracting content embeddings from transcripts and readings...")
        
        def get_file_embedding(file_path, is_reading: bool = False) -> np.ndarray:
            try:
                if is_reading:
                    content = list(file_path.values())[0]
                else:
                    # Process transcript file
                    content = self.read_transcript_file(file_path)
                
                if not content.strip():
                    return np.zeros(768)
                
                # Create chunks for embedding
                chunks = [content[i:i+500] for i in range(0, len(content), 500)][:10]
                chunk_embeddings = self.sentence_model.encode(chunks)
                return np.mean(chunk_embeddings, axis=0)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                return np.zeros(768)
        
        # Process transcript files
        transcript_embeddings = Parallel(n_jobs=-1, verbose=1)(
            delayed(get_file_embedding)(fp, False) for fp in file_paths
        )
        
        # Process reading files
        reading_embeddings = Parallel(n_jobs=-1, verbose=1)(
            delayed(get_file_embedding)(fp, True) for fp in reading_contents
        )
        
        # Combine embeddings
        all_embeddings = transcript_embeddings + reading_embeddings
        return all_embeddings

    def cluster_content_topics_hdbscan_with_readings(self, embeddings: List[np.ndarray],
                                                file_paths: List[str],
                                                reading_contents) -> Dict[int, Dict[str, List[str]]]:
        """Cluster content topics including both transcripts and readings"""
        reading_file_paths = [list(each.keys())[0] for each in reading_contents]
        if not embeddings or len(embeddings) < 2:
            return {0: {"transcripts": file_paths, "readings": reading_file_paths}}
        
        embeddings_array = np.array(embeddings)
        min_cluster_size = max(2, len(embeddings) // 4)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            allow_single_cluster=False
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_array)
        
        # Separate transcript and reading files
        all_files = file_paths + reading_file_paths
        file_types = ['transcript'] * len(file_paths) + ['reading'] * len(reading_file_paths)
        
        clusters = defaultdict(lambda: {"transcripts": [], "readings": []})
        
        for file_path, file_type, cluster_id in zip(all_files, file_types, cluster_labels):
            if file_type == 'transcript':
                clusters[cluster_id]["transcripts"].append(file_path)
            else:
                clusters[cluster_id]["readings"].append(file_path)
        
        # Handle noise points
        if -1 in clusters:
            max_cluster_id = max([cid for cid in clusters.keys() if cid != -1], default=-1)
            clusters[max_cluster_id + 1] = clusters.pop(-1)
        
        return dict(clusters)
    


    def analyze_cluster_characteristics_with_readings(self, clusters: Dict[int, Dict[str, List[str]]],
                                                    course_info: str, difficulty_level: str,
                                                    learning_objectives: List[str], reading_contents_helper) -> List[TopicCluster]:
        """Analyze cluster characteristics including reading materials"""
        cluster_characteristics = []
        service = get_gdrive_service(self.service_account_file)
        
        for cluster_id, cluster_files in clusters.items():
            sample_content = []
            
            # Process transcript files
            for fp in cluster_files.get("transcripts", []):
                content = self.read_transcript_file(fp)
                if content:
                    sample_content.append(f"TRANSCRIPT: {content[:600]}")
            
            # Process reading files
            for fp in cluster_files.get("readings", []):
                content = reading_contents_helper[fp]
                if content:
                    sample_content.append(f"READING: {content[:600]}")
            
            combined_sample = "\n\n".join(sample_content)
            
            prompt = f"""Analyze this content cluster containing both lecture transcripts and reading materials.

            Course: {course_info}
            Difficulty: {difficulty_level}
            Module Learning Objectives: {', '.join(learning_objectives)}

            Sample Content: {combined_sample}

            IMPORTANT: Respond ONLY with a single valid JSON object. Do not include any text before or after the JSON. Do not use markdown formatting or code blocks.

            Return exactly this JSON structure:
            {{"cluster_name": "Topic Name (2-4 words)", "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"], "importance_level": "basic", "recommended_content_ratio": 0.25}}

            Requirements:
            - cluster_name: 2-4 words describing the main topic
            - keywords: exactly 5 relevant keywords as strings
            - importance_level: must be exactly one of these values: "basic", "intermediate", "advanced", "critical"
            - recommended_content_ratio: a decimal number between 0.1 and 0.5
            - Consider both lecture content and reading materials when determining importance and ratios
            - Ensure all string values are properly quoted
            - Do not include trailing commas
            - Response must be parseable as valid JSON
            """
            
            try:
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()
                
                if not response_text:
                    raise ValueError("Empty response")
                
                if not response_text.startswith('{'):
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        response_text = response_text[json_start:json_end]
                    else:
                        raise ValueError("No JSON found in response")
                
                cluster_data = json.loads(response_text)
                
                cluster_name = cluster_data.get("cluster_name", f"Topic {cluster_id}")
                keywords = cluster_data.get("keywords", [])
                importance_level = cluster_data.get("importance_level", "intermediate")
                recommended_ratio = cluster_data.get("recommended_content_ratio", 0.25)
                
                if importance_level not in ["basic", "intermediate", "advanced", "critical"]:
                    importance_level = "intermediate"
                
                if not isinstance(recommended_ratio, (int, float)) or recommended_ratio < 0.1 or recommended_ratio > 0.5:
                    recommended_ratio = 0.25
                
                cluster_characteristics.append(TopicCluster(
                    cluster_id=cluster_id,
                    cluster_name=cluster_name,
                    keywords=keywords if isinstance(keywords, list) else [],
                    importance_level=importance_level,
                    recommended_content_ratio=float(recommended_ratio)
                ))
                
            except Exception as e:
                logger.error(f"Error analyzing cluster {cluster_id}: {e}")
                
                fallback_name = f"Content Area {cluster_id + 1}"
                if sample_content:
                    words = combined_sample.lower().split()[:50]
                    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                    meaningful_words = [w for w in words if len(w) > 3 and w not in common_words]
                    if meaningful_words:
                        fallback_name = f"{meaningful_words[0].title()} Topics"
                
                cluster_characteristics.append(TopicCluster(
                    cluster_id=cluster_id,
                    cluster_name=fallback_name,
                    keywords=[],
                    importance_level="intermediate",
                    recommended_content_ratio=0.25
                ))
        
        return cluster_characteristics

    def calculate_content_distribution_with_readings(self, clusters: Dict[int, Dict[str, List[str]]],
                                                file_paths: List[str],
                                                reading_contents) -> Dict[int, float]:
        """Calculate content distribution including reading materials"""
        reading_file_paths = [list(each.keys())[0] for each in reading_contents]
        reading_file_contents = [list(each.values())[0] for each in reading_contents]

        cluster_content_ratios = defaultdict(float)
        service = get_gdrive_service(self.service_account_file)
        
        total_words = 0
        file_word_counts = {}
        
        # Count words in transcript files
        for fp in file_paths:
            content = self.read_transcript_file(fp)
            word_count = len(content.split()) if content else 0
            file_word_counts[fp] = word_count
            total_words += word_count
        
        # Count words in reading files
        for each in reading_contents:
            fp = list(each.keys())[0]
            content = list(each.values())[0]
            word_count = len(content.split()) if content else 0
            file_word_counts[fp] = word_count
            total_words += word_count
        
        if total_words == 0:
            return dict(cluster_content_ratios)
        
        for cluster_id, cluster_files in clusters.items():
            cluster_words = 0
            
            # Add transcript words
            for fp in cluster_files.get("transcripts", []):
                cluster_words += file_word_counts.get(fp, 0)
            
            # Add reading words
            for fp in cluster_files.get("readings", []):
                cluster_words += file_word_counts.get(fp, 0)
            
            cluster_content_ratios[cluster_id] = cluster_words / total_words
        
        return dict(cluster_content_ratios)
    
    def calculate_tdb_score(self, cluster_characteristics: List[TopicCluster],
                          actual_content_distribution: Dict[int, float]) -> TDBResult:
        """Calculate Topic Distribution Balance score"""
        cluster_analysis = {}
        imbalance_penalties = {}
        recommendations = []

        for cluster in cluster_characteristics:
            actual_ratio = actual_content_distribution.get(cluster.cluster_id, 0)
            recommended_ratio = cluster.recommended_content_ratio
            deviation = abs(actual_ratio - recommended_ratio)

            importance_weights = {
                'basic': 1.0,
                'intermediate': 1.2,
                'advanced': 1.5,
                'critical': 2.0
            }

            weight = importance_weights.get(cluster.importance_level, 1.0)
            penalty = deviation * weight

            cluster_analysis[cluster.cluster_name] = {
                'actual_content_ratio': actual_ratio,
                'recommended_content_ratio': recommended_ratio,
                'deviation': deviation,
                'importance_level': cluster.importance_level,
                'penalty': penalty
            }

            imbalance_penalties[cluster.cluster_name] = penalty

            if deviation > 0.15:
                if actual_ratio > recommended_ratio:
                    recommendations.append(
                        f"Reduce content focus on '{cluster.cluster_name}' by {(deviation*100):.1f}%"
                    )
                else:
                    recommendations.append(
                        f"Increase content coverage of '{cluster.cluster_name}' by {(deviation*100):.1f}%"
                    )

        total_penalty = sum(imbalance_penalties.values())
        max_possible_penalty = len(cluster_characteristics) * 2.0

        if max_possible_penalty > 0:
            balance_score = max(0, 1 - (total_penalty / max_possible_penalty))
        else:
            balance_score = 1.0

        return TDBResult(
            tdb_score=balance_score,
            cluster_analysis=cluster_analysis,
            content_distribution=actual_content_distribution,
            imbalance_penalties=imbalance_penalties,
            recommendations=recommendations
        )

    # Dynamic RWSS Implementation
    def download_pdf_from_gdrive_1(self, file_id: str, service) -> str:
        """Download PDF from Google Drive to temporary storage"""
        try:
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name', f'{file_id}.pdf')
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_path = temp_file.name
            temp_file.close()
            
            # Download file content
            request = service.files().get_media(fileId=file_id)
            with open(temp_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
            
            return temp_path
        except Exception as e:
            logger.error(f"Error downloading PDF {file_id}: {e}")
            return None

    def process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file for Gemini without altering format"""
        try:
            with open(file_path, "rb") as pdf_file:
                return {
                    "mime_type": "application/pdf",
                    "data": pdf_file.read(),
                }
        except Exception as e:
            logger.error(f"Failed to read PDF file {file_path}: {e}")
            return None

    def detect_real_world_scenarios_from_pdf(self, file_id: str, service) -> List[RealWorldScenario]:
        """Detect real-world scenarios from PDF files using Gemini"""
        temp_pdf_path = None
        try:
            # Download PDF to temporary storage
            temp_pdf_path = self.download_pdf_from_gdrive_1(file_id, service)
            if not temp_pdf_path:
                return []
            
            # Process PDF file for Gemini
            pdf_data = self.process_pdf_file(temp_pdf_path)
            if not pdf_data:
                return []
            
            # Enhanced prompt for PDF analysis
            prompt = f"""
            Analyze this PDF document comprehensively and identify real-world scenarios, practical applications, and industry relevance.
            
            Look for:
            1. Case studies or real business examples
            2. Practical projects or hands-on exercises
            3. Industry-specific applications
            4. Real datasets or data sources mentioned
            5. Professional tools and technologies
            6. Implementation examples
            7. Commercial or enterprise use cases
            8. Any practical applications of concepts
            
            Be very generous in identifying educational content that has real-world value.
            Even theoretical concepts often have practical implications.
            
            Return a JSON array with this exact structure:
            [
                {{
                    "scenario_type": "case_study|project|dataset|industry_application|practical_application",
                    "content": "relevant text or description from the document",
                    "confidence": 0.6,
                    "industry_domain": "finance|healthcare|technology|education|general|etc",
                    "detected_tools": ["tool1", "tool2", "framework1"],
                    "page_reference": "page number or section if identifiable"
                }}
            ]
            
            Return empty array [] if no significant real-world indicators are found.
            """
            
            # Initialize Gemini model
            model = genai.GenerativeModel("models/gemini-1.5-flash", 
                                        generation_config={"temperature": 0.0})
            
            # Send PDF and prompt to Gemini
            message_parts = [prompt, pdf_data]
            response = model.generate_content(message_parts)
            response_text = response.text.strip()
            
            # Parse response
            if response_text.startswith("```"):
                response_text = response_text.lstrip("```json").rstrip("```")
            
            # Extract JSON array
            match = re.search(r'$$\s*{.*?}\s*]', response_text, re.DOTALL)
            if match:
                json_text = match.group(0)
                pdf_scenarios_data = json.loads(json_text)
            else:
                pdf_scenarios_data = json.loads(response_text) if response_text.startswith('[') else []
            
            # Convert to RealWorldScenario objects
            scenarios = []
            for scenario_data in pdf_scenarios_data:
                if isinstance(scenario_data, dict) and scenario_data.get('confidence', 0) >= 0.2:
                    # Apply generous boost for PDF content
                    boosted_confidence = min(scenario_data.get('confidence', 0.3) * 1.3 + 0.15, 1.0)
                    
                    scenarios.append(RealWorldScenario(
                        scenario_type=scenario_data.get('scenario_type', 'practical_application'),
                        content=scenario_data.get('content', ''),
                        confidence_score=boosted_confidence,
                        file_source=f"pdf_{file_id}",
                        industry_domain=scenario_data.get('industry_domain', 'general'),
                        detected_tools=scenario_data.get('detected_tools', [])
                    ))
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_id}: {e}")
            return []
        finally:
            # Clean up temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except:
                    pass
    
    def _detect_ai_real_world_indicators(self, content: str, file_path: str) -> List[RealWorldScenario]:
        """Use AI to detect additional real-world indicators"""
        # Sample content for AI analysis
        content_sample = content

        prompt = f"""
        Analyze this educational content and identify segments that indicate real-world applications,
        practical scenarios, or industry relevance.

        Content: {content_sample}

        Look for:
        - Mentions of real companies or organizations
        - Practical examples or use cases
        - Industry-specific terminology
        - References to real datasets or projects
        - Professional tools or workflows

        Return a JSON list of detected real-world segments with this structure:
        [
            {{
                "type": "industry_example|practical_application|real_dataset|professional_tool",
                "content": "the relevant text segment",
                "confidence": 0.8,
                "tools": ["tool1", "tool2"],
                "domain": "finance|healthcare|etc"
            }}
        ]

        Return empty list [] if no significant real-world indicators are found.
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            print("Generated Gemini Content..............")
            ai_scenarios = json.loads(response.text.strip())

            scenarios = []
            for scenario_data in ai_scenarios:
                if isinstance(scenario_data, dict) and scenario_data.get('confidence', 0) >= 0.6:
                    scenarios.append(RealWorldScenario(
                        scenario_type=scenario_data.get('type', 'real_application'),
                        content=scenario_data.get('content', ''),
                        confidence_score=scenario_data.get('confidence', 0.6),
                        file_source=file_path,
                        industry_domain=scenario_data.get('domain', 'general'),
                        detected_tools=scenario_data.get('tools', [])
                    ))

            return scenarios
        except Exception as e:
            logger.error(f"Error in AI real-world detection: {e}")
            return []

    def detect_real_world_scenarios_per_file(self, file_path: str) -> List[RealWorldScenario]:
        """Enhanced version - detect real-world scenarios with much more lenient thresholds"""
        content = self.read_transcript_file(file_path)
        if not content:
            return []

        scenarios = []
        content_lower = content.lower()

        # Use AI-generated patterns with very lenient detection
        for scenario_type, patterns in self.scenario_patterns.items():
            for pattern in patterns:
                pattern_regex = r'\b' + re.escape(pattern.lower()) + r'\b'
                matches = re.finditer(pattern_regex, content_lower, re.IGNORECASE)

                for match in matches:
                    start = max(0, match.start() - 150)
                    end = min(len(content), match.end() + 150)
                    context = content[start:end]

                    # Much more lenient confidence threshold
                    confidence = self._calculate_ai_scenario_confidence(context, scenario_type)

                    if confidence >= 0.2:  # Reduced from 0.5 to 0.2
                        industry_domain = self._detect_ai_industry_domain(context)
                        detected_tools = self._detect_ai_tools(context)

                        scenarios.append(RealWorldScenario(
                            scenario_type=scenario_type,
                            content=context.strip(),
                            confidence_score=confidence,
                            file_source=file_path,
                            industry_domain=industry_domain,
                            detected_tools=detected_tools
                        ))

        # Additional AI-based detection with much lower threshold
        additional_scenarios = self._detect_ai_real_world_indicators(content, file_path)
        scenarios.extend(additional_scenarios)

        return scenarios

    def process_real_world_scenarios_parallel(self, file_paths: List[str], 
                                            reading_file_paths) -> List[RealWorldScenario]:
        """Enhanced version - process both transcript and PDF files in parallel"""
        logger.info("Detecting real-world scenarios using AI across all files...")
        
        all_scenarios = []
        
        # Process transcript files
        if file_paths:
            scenario_lists = Parallel(n_jobs=-1, verbose=1)(
                delayed(self.detect_real_world_scenarios_per_file)(fp) for fp in file_paths
            )
            
            # Flatten transcript scenarios
            for scenario_list in scenario_lists:
                all_scenarios.extend(scenario_list)
        
        # Process PDF files if provided
        if reading_file_paths:
            service = get_gdrive_service(self.service_account_file)
            
            # Process PDFs in parallel
            pdf_scenario_lists = Parallel(n_jobs=-1, verbose=1)(
                delayed(self.detect_real_world_scenarios_from_pdf)(file_id, service) 
                for file_id in reading_file_paths
            )
            
            # Flatten PDF scenarios
            for scenario_list in pdf_scenario_lists:
                all_scenarios.extend(scenario_list)
        
        # Deduplicate scenarios
        unique_scenarios = []
        seen_scenarios = set()
        
        for scenario in all_scenarios:
            # Create a hash for deduplication based on content similarity
            scenario_hash = hash((scenario.scenario_type, scenario.content[:100], 
                                scenario.industry_domain))
            if scenario_hash not in seen_scenarios:
                seen_scenarios.add(scenario_hash)
                unique_scenarios.append(scenario)
        
        return unique_scenarios

    def calculate_rwss_score_enhanced(self, scenarios: List[RealWorldScenario],
                                    module_input: ModuleInput) -> RWSSResult:
        """Enhanced RWSS calculation with PDF content consideration"""
        if not scenarios:
            return RWSSResult(0.0, [], {}, {}, 0.0, 0.0)

        # Separate PDF and transcript scenarios for analysis
        pdf_scenarios = [s for s in scenarios if s.file_source.startswith('pdf_')]
        transcript_scenarios = [s for s in scenarios if not s.file_source.startswith('pdf_')]
        
        # Use very inclusive confidence thresholds
        moderate_confidence_scenarios = [s for s in scenarios if s.confidence_score >= 0.2]
        high_confidence_scenarios = [s for s in scenarios if s.confidence_score >= 0.4]
        
        # Calculate scenario distribution
        scenario_distribution = Counter(s.scenario_type for s in moderate_confidence_scenarios)
        
        # Calculate industry coverage
        industry_coverage = Counter(s.industry_domain for s in moderate_confidence_scenarios)
        total_scenarios = len(moderate_confidence_scenarios)
        industry_coverage_ratio = {k: v/total_scenarios for k, v in industry_coverage.items()} if total_scenarios > 0 else {}
        
        # Enhanced tool diversity score with PDF bonus
        unique_tools = set()
        for scenario in moderate_confidence_scenarios:
            unique_tools.update(scenario.detected_tools)
        
        tool_count = len(unique_tools)
        if tool_count == 0:
            tool_diversity_score = 0.2
        elif tool_count == 1:
            tool_diversity_score = 0.5
        elif tool_count == 2:
            tool_diversity_score = 0.7
        elif tool_count == 3:
            tool_diversity_score = 0.85
        elif tool_count <= 4:
            tool_diversity_score = 0.95
        else:
            tool_diversity_score = 1.0
        
        # Enhanced practical application score
        practical_scenarios = [s for s in moderate_confidence_scenarios
                            if s.scenario_type in ['project', 'case_study', 'industry_application', 'practical_application']]
        
        practical_count = len(practical_scenarios)
        if practical_count == 0:
            practical_application_score = 0.3
        elif practical_count == 1:
            practical_application_score = 0.7
        elif practical_count == 2:
            practical_application_score = 0.9
        else:
            practical_application_score = 1.0
        
        # Calculate bonuses
        scenario_types_count = len(scenario_distribution)
        scenario_variety_bonus = min(scenario_types_count * 0.4, 1.0)
        
        industry_types_count = len(industry_coverage)
        industry_variety_bonus = min(industry_types_count * 0.5, 1.0)
        
        # Base score calculation with PDF content bonus
        base_scenario_count = total_scenarios
        pdf_content_bonus = 0.1 if len(pdf_scenarios) > 0 else 0  # Bonus for having PDF content
        
        if base_scenario_count == 0:
            base_score = 0.1
        elif base_scenario_count == 1:
            base_score = 0.5
        elif base_scenario_count == 2:
            base_score = 0.7
        elif base_scenario_count == 3:
            base_score = 0.85
        elif base_scenario_count <= 4:
            base_score = 0.95
        else:
            base_score = 1.0
        
        # Quality and content bonuses
        high_confidence_ratio = len(high_confidence_scenarios) / max(total_scenarios, 1)
        quality_bonus = high_confidence_ratio * 0.25
        
        avg_confidence = np.mean([s.confidence_score for s in moderate_confidence_scenarios]) if moderate_confidence_scenarios else 0
        content_richness_bonus = min((avg_confidence - 0.2) * 0.3, 0.15) if avg_confidence > 0.2 else 0
        
        educational_bonus = 0.1 if total_scenarios > 0 else 0
        
        # Enhanced weighting with PDF consideration
        rwss_score = (0.40 * base_score +
                    0.15 * tool_diversity_score +
                    0.15 * practical_application_score +
                    0.08 * scenario_variety_bonus +
                    0.07 * industry_variety_bonus +
                    0.05 * quality_bonus +
                    0.03 * content_richness_bonus +
                    0.02 * educational_bonus +
                    0.05 * pdf_content_bonus)  # New PDF bonus
        
        # Apply overall boost for educational content
        rwss_score = min(rwss_score * 1.2 + 0.1, 1.0)
        
        # Ensure minimum reasonable score
        if total_scenarios > 0:
            rwss_score = max(rwss_score, 0.3)
        
        rwss_score = min(max(rwss_score, 0.0), 1.0)
        
        return RWSSResult(
            rwss_score=rwss_score,
            detected_scenarios=moderate_confidence_scenarios,
            scenario_distribution=dict(scenario_distribution),
            industry_coverage=industry_coverage_ratio,
            tool_diversity_score=tool_diversity_score,
            practical_application_score=practical_application_score
        )

    def normalize_scores_to_1_5_scale(self, cccs_score: float, tdb_score: float, rwss_score: float) -> Tuple[float, float, float]:
        """Normalize all three scores to 1-5 scale and round to two decimal places"""
        def _normalize_and_round(score: float) -> float:
            normalized = 1.0 + (score * 4.0)  # Scale from [0,1] to [1,5]
            clamped = max(1.0, min(5.0, normalized))  # Clamp between 1.0 and 5.0
            return round(clamped, 2)  # Round to 2 decimal places

        normalized_cccs = _normalize_and_round(cccs_score)
        normalized_tdb = _normalize_and_round(tdb_score)
        normalized_rwss = _normalize_and_round(rwss_score)

        return normalized_cccs, normalized_tdb, normalized_rwss

    def analyze_module_comprehensive(self, module_input: ModuleInput):
        """Perform comprehensive analysis with dynamically generated patterns"""
        logger.info("Starting comprehensive module analysis with dynamic AI patterns...")

        # Step 1: Generate course-specific patterns using AI
        self.generate_course_specific_patterns(
            module_input.course_info,
            module_input.difficulty_level,
            module_input.learning_objectives
        )

        # Step 2: Calculate CCCS
        cccs_result = self.calculate_cccs_full(module_input)

        # Step 3: Calculate TDB
        embeddings = self.extract_content_embeddings_with_readings(
            module_input.transcript_file_paths, 
            module_input.reading_contents
        )

        clusters = self.cluster_content_topics_hdbscan_with_readings(
            embeddings, 
            module_input.transcript_file_paths,
            module_input.reading_contents
        )

        reading_contents_helper = {k: v for d in module_input.reading_contents for k, v in d.items()}

        cluster_characteristics = self.analyze_cluster_characteristics_with_readings(
            clusters, 
            module_input.course_info, 
            module_input.difficulty_level, 
            module_input.learning_objectives,
            reading_contents_helper
        )

        content_distribution = self.calculate_content_distribution_with_readings(
            clusters, 
            module_input.transcript_file_paths,
            module_input.reading_contents
        )

        tdb_result = self.calculate_tdb_score(cluster_characteristics, content_distribution)

        # Step 4: Calculate RWSS using dynamic patterns
        reading_file_paths = [list(each.keys())[0] for each in module_input.reading_contents]
        scenarios = self.process_real_world_scenarios_parallel(
            module_input.transcript_file_paths, 
            reading_file_paths  # Pass the PDF file IDs
        )
        rwss_result = self.calculate_rwss_score_enhanced(scenarios, module_input)

        # Step 5: Normalize scores to 1-5 scale
        normalized_cccs, normalized_tdb, normalized_rwss = self.normalize_scores_to_1_5_scale(
            cccs_result.final_score,
            tdb_result.tdb_score,
            rwss_result.rwss_score
        )


        logger.info(f"Dynamic analysis complete. CCCS: {normalized_cccs:.2f}, TDB: {normalized_tdb:.2f}, RWSS: {normalized_rwss:.2f}")
        
        return {
            "normalized_cccs_score": normalized_cccs,
            "normalized_tdb_score": normalized_tdb,
            "normalized_rwss_score": normalized_rwss,
            "cccs_result": cccs_result,
            "tdb_result": tdb_result,
            "rwss_result": rwss_result
        }


class MinimalPADCalculator:
    def __init__(self, gemini_api_key: str, service_account_file: str):
        """Initialize with GenAI for minimal keyword generation"""
        genai.configure(api_key=gemini_api_key) # Use the passed API key
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.service_account_file = service_account_file
        self.practical_indicators = []
        self.theoretical_indicators = []
        self.model_for_vision = genai.GenerativeModel('gemini-2.0-flash')
        logging.info("MinimalPADCalculator initialized.")

    def _generate_minimal_keywords(self, course_info, difficulty_level, learning_objectives):
        """Generate minimal keyword sets with detailed explanations and strict output format."""
        context = f"""
        Course: {course_info}
        Difficulty: {difficulty_level}
        Learning Objectives: {'; '.join(learning_objectives)}
        """

        prompt = f"""
        Based on the following course context, generate exactly 10 practical keywords and 5 theoretical keywords that are most indicative of hands-on vs conceptual content.

        For each keyword, provide a one-sentence explanation of why it is relevant to the course's practical or theoretical aspects.

        Return your answer in this exact JSON format:
        {{
          "practical": [
            {{"keyword": "keyword1", "explanation": "..." }},
            ...
            {{"keyword": "keyword10", "explanation": "..." }}
          ],
          "theoretical": [
            {{"keyword": "keyword1", "explanation": "..." }},
            ...
            {{"keyword": "keyword5", "explanation": "..." }}
          ]
        }}

        Only output valid JSON. Do not include any extra text or commentary.
        Context:
        {context}
        """

        try:
            response = self.model.generate_content(prompt)
            self._parse_minimal_keywords_json(response.text)
        except Exception as e:
            print(f"Keyword generation failed: {e}")
            # Fallback
            self.practical_indicators = ['implement', 'build', 'create', 'develop', 'apply', 'practice', 'exercise', 'project', 'lab', 'demo']
            self.theoretical_indicators = ['theory', 'concept', 'principle', 'framework', 'overview']

    def _parse_minimal_keywords_json(self, text):
        """Parse minimal keyword sets from strict JSON GenAI response."""
        try:
            import json
            data = json.loads(text)
            self.practical_indicators = [item['keyword'] for item in data.get('practical', [])]
            self.theoretical_indicators = [item['keyword'] for item in data.get('theoretical', [])]
        except Exception:
            # Fallback
            self.practical_indicators = ['implement', 'build', 'create', 'develop', 'apply', 'practice', 'exercise', 'project', 'lab', 'demo']
            self.theoretical_indicators = ['theory', 'concept', 'principle', 'framework', 'overview']

    def download_file_bytes_and_mime_type(self, file_id: str) -> Tuple[bytes, str]:
        """
        Downloads a file from Google Drive using the file ID and returns its bytes and MIME type.
        This method replaces the previous internal _download_file_bytes_and_mime_type.

        Args:
            file_id: ID of the file to download.

        Returns:
            Tuple containing the file's bytes and its MIME type.
        """
        service = get_gdrive_service(self.service_account_file) # Build service internally
        try:
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id, fields='mimeType, name').execute()
            mime_type = file_metadata.get('mimeType', 'application/octet-stream')
            file_name = file_metadata.get('name', 'file') # For logging/debugging if needed

            # Download file content
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            fh.seek(0)
            file_bytes = fh.read()

            return file_bytes, mime_type

        except Exception as e:
            logging.error(f"Error downloading file {file_id}: {e}")
            raise # Re-raise to be caught by the calling function for r

    def convert_odt_to_pdf(self, input_bytes: bytes, output_dir: str = None):
        """
        Converts .odt content (as bytes) to PDF bytes using LibreOffice.
        Creates a temporary file for conversion.

        Args:
            input_bytes (bytes): The byte content of the .odt file.
            output_dir (str, optional): Directory to save the output PDF. If None,
                                        a temporary directory will be used.

        Returns:
            Optional[bytes]: The byte content of the generated PDF file, or None if conversion fails or results in an invalid PDF.
        """
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir = output_dir

        temp_input_path = os.path.join(temp_dir, "temp_input.odt")
        temp_output_pdf_path = os.path.join(temp_dir, "temp_input.pdf")
        pdf_bytes = None # Initialize to None

        try:
            with open(temp_input_path, "wb") as f:
                f.write(input_bytes)

            # NOTE: Path to soffice.exe is hardcoded for Windows.
            # For Linux/macOS, it might be just 'soffice' if it's in PATH, or a specific path like '/usr/bin/soffice'.
            # Make sure LibreOffice is installed and accessible in the environment where this code runs.
            subprocess_result = subprocess.run([
                r"/usr/bin/soffice", # Changed to 'soffice' for broader compatibility (assuming it's in PATH)
                "--headless",
                "--convert-to", "pdf",
                "--outdir", temp_dir,
                temp_input_path
            ], capture_output=True, text=True) # Capture output for better error reporting

            if subprocess_result.returncode != 0:
                logging.error(f"LibreOffice conversion failed with exit code {subprocess_result.returncode}.")
                logging.error(f"Stdout: {subprocess_result.stdout}")
                logging.error(f"Stderr: {subprocess_result.stderr}")
                return None # Conversion failed

            if not os.path.exists(temp_output_pdf_path) or os.path.getsize(temp_output_pdf_path) == 0:
                logging.error(f"PDF was not created or is empty after soffice conversion. soffice stdout: {subprocess_result.stdout}, stderr: {subprocess_result.stderr}")
                return None

            with open(temp_output_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # **New: Validate if the generated bytes are actually a PDF**
            try:
                # Attempt to read with PdfReader. If this fails, it's not a valid PDF.
                reader = PdfReader(BytesIO(pdf_bytes))
                # Optionally, iterate through pages to force full parsing if needed
                # for page in reader.pages:
                #     _ = page.extract_text()
                logging.info(f"Successfully converted ODT to PDF for {temp_input_path}")
                return pdf_bytes
            except Exception as e:
                logging.error(f"Generated file is not a valid PDF for {temp_input_path}: {e}")
                logging.error(f"Problematic PDF bytes start: {pdf_bytes[:50]}") # Log first 50 bytes for debugging
                return None # Not a valid PDF

        except FileNotFoundError:
            logging.error("LibreOffice (soffice) not found. Please ensure it's installed and its path is correct in your system's PATH or hardcoded if necessary.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during ODT to PDF conversion: {e}")
            return None
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_pdf_path):
                os.remove(temp_output_pdf_path)
            if output_dir is None and os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError as e:
                    logging.error(f"Error removing temporary directory {temp_dir}: {e}")

    def read_odt_file(self, file_id, max_retries=5):
        """Reads content from a Google Drive ODT file by converting it to PDF first."""
        service = get_gdrive_service(self.service_account_file)
        retries = 0

        while retries < max_retries:
            try:
                # Get the ODT file content
                odt_bytes = service.files().get_media(fileId=file_id).execute()

                # Convert ODT to PDF
                pdf_bytes = self.convert_odt_to_pdf(odt_bytes)

                if pdf_bytes is None: # This check is crucial now
                    logging.error(f"Failed to convert ODT file {file_id} to valid PDF. Returning empty content.")
                    return ""

                # Extract text from PDF
                reader = PdfReader(BytesIO(pdf_bytes))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                # Clean up the extracted text
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            except (HttpError, http.client.IncompleteRead) as e:
                retries += 1
                logging.warning(f"Attempt {retries} failed for ODT file {file_id}: {str(e)}")
                if retries >= max_retries:
                    logging.error(f"Failed to read ODT file {file_id} after {max_retries} attempts: {str(e)}")
                    return ""
                # Exponential backoff
                wait_time = 2 ** retries
                time.sleep(wait_time)
        return ""


    def read_text_file(self, file_id, max_retries=5):
        """Reads content from a Google Drive text file."""
        service = get_gdrive_service(self.service_account_file) # Fixed: pass the parameter
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
                logging.warning(f"Attempt {retries} failed for text file {file_id}: {str(e)}")
                if retries >= max_retries:
                    logging.error(f"Failed to read file {file_id} after {max_retries} attempts: {str(e)}")
                    return ""
                # Exponential backoff
                wait_time = 2 ** retries
                time.sleep(wait_time)
        return "" # Should not be reached, but for safety

    def read_pdf_file(self, file_id, max_retries=5):
        """Reads content from a Google Drive PDF file."""
        service = get_gdrive_service(self.service_account_file) # Fixed: pass the parameter
        retries = 0

        while retries < max_retries:
            try:
                # In-memory fetch from Google Drive
                pdf_bytes = service.files().get_media(fileId=file_id).execute()

                # In-memory PDF reader
                reader = PdfReader(BytesIO(pdf_bytes))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

                # Clean up the extracted text
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            except (HttpError, http.client.IncompleteRead) as e:
                retries += 1
                logging.warning(f"Attempt {retries} failed for PDF file {file_id}: {str(e)}")
                if retries >= max_retries:
                    logging.error(f"Failed to read PDF {file_id} after {max_retries} attempts: {str(e)}")
                    return ""
                wait_time = 2 ** retries
                logging.info(f"Attempt {retries} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        return "" # Should not be reached, but for safety

    def advanced_ml_analysis(self, content):
        """Apply advanced NLP techniques to analyze content for practical vs theoretical orientation."""
        if not content or len(content.strip()) < 50:
            return self._keyword_helper_score(content)

        try:
            # Improved chunking: use sentence transformers for semantic chunking if available
            try:
                time.sleep(5)
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                sentences = re.split(r'(?<=[.!?]) +', content)
                embeddings = model.encode(sentences)
                # Cluster sentences into chunks based on semantic similarity
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(sentences) // 10 + 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                chunks = [' '.join([sentences[i] for i in range(len(sentences)) if labels[i] == c]) for c in range(n_clusters)]
            except Exception:
                # Fallback to original chunking
                chunks = self._create_content_chunks(content)

            ml_scores = []

            # 1. TF-IDF with n-grams
            for ngram_range in [(1, 2), (1, 3), (2, 4)]:
                try:
                    vectorizer = TfidfVectorizer(
                        max_features=min(5000, len(chunks) * 20),
                        stop_words='english',
                        ngram_range=ngram_range,
                        min_df=1,
                        max_df=0.95,
                        sublinear_tf=True,
                        lowercase=True
                    )
                    X = vectorizer.fit_transform(chunks)
                    score = self._analyze_tfidf_matrix(X, chunks, vectorizer)
                    ml_scores.append(score)
                except Exception:
                    continue

            # 2. Topic modeling (LDA)
            try:
                from sklearn.decomposition import LatentDirichletAllocation
                vectorizer = TfidfVectorizer(stop_words='english')
                X = vectorizer.fit_transform(chunks)
                lda = LatentDirichletAllocation(n_components=2, random_state=42)
                lda.fit(X)
                # Score based on topic-word distribution overlap with indicators
                feature_names = vectorizer.get_feature_names_out()
                topics = lda.components_
                practical_score = sum(topics[0][i] for i, f in enumerate(feature_names) if any(ind in f for ind in self.practical_indicators))
                theoretical_score = sum(topics[1][i] for i, f in enumerate(feature_names) if any(ind in f for ind in self.theoretical_indicators))
                total = practical_score + theoretical_score
                if total > 0:
                    ml_scores.append((practical_score / total) * 100)
            except Exception:
                pass

            # 3. Embedding-based similarity (optional, if sentence-transformers is available)
            # ... (additional advanced techniques can be added here)

            # Combine ML scores with keyword helper
            if ml_scores:
                ml_average = np.mean(ml_scores)
                keyword_score = self._keyword_helper_score(content)
                final_score = 0.8 * ml_average + 0.2 * keyword_score
                return final_score
            else:
                return self._keyword_helper_score(content)

        except Exception:
            return self._keyword_helper_score(content)


    def get_pdf_summary_with_gemini(self, file_bytes: bytes, mime_type: str) -> str:
        """
        Gets a comprehensive summary of a PDF file using Gemini.
        This summary will include text content and insights from images.
        """
        prompt = (
            "Summarize the attached document comprehensively. Include all key information, "
            "main arguments, and any relevant details from both text and images. "
            "Focus on providing a detailed overview of the content's purpose and key takeaways. "
            "The summary should be extensive enough to capture the essence for further analysis."
        )

        file_part = {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(file_bytes).decode('utf-8')
                    }
                }
            ]
        }

        try:
            # Use the vision model for summarizing multimodal content
            response = self.model_for_vision.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": prompt}]},
                    file_part
                ]
            )
            return response.text
        except Exception as e:
            logging.error(f"Error generating PDF summary with Gemini: {e}")
            return "" # Return empty string on failure

    def _create_content_chunks(self, content, min_chunk_size=200):
        """Create semantically meaningful chunks using sentence boundaries."""
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(content)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)
            if current_length >= min_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]


    def _analyze_tfidf_matrix(self, X, chunks, vectorizer):
        """Analyze TF-IDF matrix for practical orientation"""
        try:
            feature_names = vectorizer.get_feature_names_out()

            # Calculate practical vs theoretical feature importance
            practical_features = []
            theoretical_features = []

            for i, feature in enumerate(feature_names):
                for indicator in self.practical_indicators:
                    if indicator.lower() in feature.lower():
                        practical_features.append(i)
                        break

                for indicator in self.theoretical_indicators:
                    if indicator.lower() in feature.lower():
                        theoretical_features.append(i)
                        break

            if not practical_features and not theoretical_features:
                return 50.0  # Neutral if no indicators found

            # Calculate weighted scores
            X_dense = X.toarray()
            practical_weights = np.sum(X_dense[:, practical_features], axis=1) if practical_features else np.zeros(X_dense.shape[0])
            theoretical_weights = np.sum(X_dense[:, theoretical_features], axis=1) if theoretical_features else np.zeros(X_dense.shape[0])

            # Calculate score for each document/chunk
            scores_per_chunk = []
            for i in range(len(chunks)):
                total_weight = practical_weights[i] + theoretical_weights[i]
                if total_weight > 0:
                    scores_per_chunk.append((practical_weights[i] / total_weight) * 100)
                else:
                    scores_per_chunk.append(50.0) # Neutral if chunk has no relevant keywords

            return np.mean(scores_per_chunk) if scores_per_chunk else 50.0

        except Exception as e:
            logging.error(f"Error in _analyze_tfidf_matrix: {e}")
            return 50.0

    def _dimensionality_analysis(self, chunks, method='pca'):
        """Apply dimensionality reduction and analyze results"""
        if len(chunks) < 2:
            return 50.0
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(2000, len(chunks) * 15),
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )

            X = vectorizer.fit_transform(chunks)
            n_components = min(50, X.shape[1]-1, X.shape[0]-1)

            if n_components <= 0:
                return 50.0 # Not enough data for meaningful reduction

            X_reduced = None
            if method == 'pca':
                reducer = PCA(n_components=n_components)
                X_reduced = reducer.fit_transform(X.toarray())
            elif method == 'svd':
                reducer = TruncatedSVD(n_components=n_components)
                X_reduced = reducer.fit_transform(X)
            elif method == 'umap' and X.shape[0] > 10: # UMAP needs more samples
                reducer = umap.UMAP(n_components=min(20, n_components), random_state=42)
                X_reduced = reducer.fit_transform(X.toarray())
            else:
                return 50.0 # Unsupported method or insufficient data for UMAP

            # Analyze reduced space for practical patterns
            return self._analyze_reduced_space(X_reduced, chunks)

        except Exception as e:
            logging.error(f"Error in _dimensionality_analysis ({method}): {e}")
            return 50.0

    def _analyze_reduced_space(self, X_reduced, chunks):
        """Analyze reduced dimensional space"""
        if X_reduced is None or X_reduced.shape[0] == 0:
            return 50.0
        try:
            # Calculate variance in each dimension
            variances = np.var(X_reduced, axis=0)

            # Use top variance dimensions to separate practical vs theoretical
            top_dims = np.argsort(variances)[-min(3, X_reduced.shape[1]):]  # Top N dimensions, ensure valid count

            practical_scores = []
            for i, chunk in enumerate(chunks):
                # Score based on position in high-variance dimensions
                dim_scores = X_reduced[i, top_dims]

                # Simple keyword check as helper
                practical_count = sum(1 for indicator in self.practical_indicators
                                     if indicator.lower() in chunk.lower())
                theoretical_count = sum(1 for indicator in self.theoretical_indicators
                                         if indicator.lower() in chunk.lower())

                # Combine dimensional analysis with keyword helpers
                keyword_ratio = 0.5 # Neutral if no keywords
                if practical_count + theoretical_count > 0:
                    keyword_ratio = practical_count / (practical_count + theoretical_count)

                # Weight dimensional position with keyword ratio
                dim_magnitude = np.mean(np.abs(dim_scores))
                # Normalize dim_magnitude to a 0-1 scale relative to the overall max magnitude
                max_overall_magnitude = np.max(np.abs(X_reduced))
                normalized_dim_magnitude = dim_magnitude / max_overall_magnitude if max_overall_magnitude > 0 else 0.5

                combined_score = 0.7 * keyword_ratio + 0.3 * normalized_dim_magnitude
                practical_scores.append(combined_score)

            return np.mean(practical_scores) * 100 if practical_scores else 50.0

        except Exception as e:
            logging.error(f"Error in _analyze_reduced_space: {e}")
            return 50.0

    def _clustering_analysis(self, chunks):
        """Apply clustering analysis"""
        if len(chunks) < 6: # HDBSCAN usually needs more samples
            return 50.0
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(1500, len(chunks) * 10),
                stop_words='english',
                ngram_range=(1, 2)
            )

            X = vectorizer.fit_transform(chunks)

            # Apply PCA for clustering
            n_components_pca = min(30, X.shape[1]-1, X.shape[0]-1)
            if n_components_pca <= 0: return 50.0 # Not enough data

            pca = PCA(n_components=n_components_pca)
            X_reduced = pca.fit_transform(X.toarray())

            # HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, len(chunks) // 4), # Adjust min_cluster_size
                metric='euclidean',
                min_samples=1 # Allow single-point clusters to be noise or small groups
            )

            clusters = clusterer.fit_predict(X_reduced)

            # Analyze each cluster
            cluster_scores = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_chunks = [chunks[i] for i in range(len(chunks)) if clusters[i] == cluster_id]
                cluster_text = ' '.join(cluster_chunks)

                cluster_score = self._keyword_helper_score(cluster_text)
                cluster_scores.append(cluster_score)

            return np.mean(cluster_scores) if cluster_scores else 50.0

        except Exception as e:
            logging.error(f"Error in _clustering_analysis: {e}")
            return 50.0

    def _keyword_helper_score(self, content):
        """Minimal keyword-based helper scoring"""
        if not content:
            return 0

        content_lower = content.lower()

        practical_count = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\w*\b', content_lower))
                             for kw in self.practical_indicators)
        theoretical_count = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\w*\b', content_lower))
                               for kw in self.theoretical_indicators)

        total = practical_count + theoretical_count
        if total > 0:
            return (practical_count / total) * 100
        else:
            return 50.0 # Changed to 50.0 (neutral) for better default if no keywords

    def process_transcript_parallel(self, file_id):
        """Process transcript files in parallel"""
        logging.info(f"Processing transcript file: {file_id}")
        content = self.read_text_file(file_id) # Should be file_id here
        if not content:
            return 0 # Or a default value indicating failure
        return self.advanced_ml_analysis(content)
    
    def process_reading_parallel(self, content):
        """Process transcript files in parallel"""
        if not content:
            return 0 # Or a default value indicating failure
        return self.advanced_ml_analysis(content)

    def calculate_pad_score(self, course_info, difficulty_level, learning_objectives, transcript_file_ids: List[str], assignment_file_ids: List[str], reading_contents):
        """Calculate PAD score with minimal keywords as helpers, including readings processed via Gemini."""

        logging.info("Calculating PAD score...")
        self._generate_minimal_keywords(course_info, difficulty_level, learning_objectives)

        transcript_scores = []
        if transcript_file_ids:
            logging.info(f"Analyzing {len(transcript_file_ids)} transcript files for PAD score.")
            transcript_scores = Parallel(n_jobs=os.cpu_count(), backend="threading")(
                delayed(self.process_transcript_parallel)(file_id) for file_id in transcript_file_ids
            )
            transcript_scores = [s for s in transcript_scores if s is not None]
            logging.info(f"Transcript PAD scores: {transcript_scores}")

        assignment_scores = []
        if assignment_file_ids:
            logging.info(f"Analyzing {len(assignment_file_ids)} assignment files (PDFs/ODTs) for PAD score.")
            for file_id in assignment_file_ids:
                content = self.read_odt_file(file_id)
                if content:
                    score = self.advanced_ml_analysis(content)
                    assignment_scores.append(score)
                else:
                    logging.warning(f"Could not read content for assignment file {file_id}. Skipping.")
            logging.info(f"Assignment PAD scores: {assignment_scores}")

        reading_scores = []
        if reading_contents:
            logging.info(f"Analyzing {len(reading_contents)} reading files (PDFs) with Gemini for PAD score.")
            for each in reading_contents:
                content = list(each.values())[0]
                reading_scores.append(self.process_reading_parallel(content))
            reading_scores = [s for s in reading_scores if s is not None]
            logging.info(f"Reading PAD scores: {reading_scores}")

        objectives_content = ' '.join(learning_objectives) if learning_objectives else ""
        objectives_score = self._keyword_helper_score(objectives_content)
        logging.info(f"Learning objectives PAD score: {objectives_score}")

        scores = []
        weights = []

        if transcript_scores:
            scores.append(np.mean(transcript_scores))
            weights.append(0.4)

        if assignment_scores:
            scores.append(np.mean(assignment_scores))
            weights.append(0.5)

        if reading_scores:
            scores.append(np.mean(reading_scores))
            weights.append(0.1)

        if objectives_score is not None:
            if not transcript_scores and not assignment_scores and not reading_scores:
                scores.append(objectives_score)
                weights.append(1.0)
            else:
                scores.append(objectives_score)
                weights.append(0.1)

        if not scores:
            logging.warning("No scores calculated for PAD. Returning 0.0.")
            return PADResult(
                final_score=0.0,
                normalized_final_score=1.0,  # Add normalized version too
                transcript_score=0.0,
                assignment_score=0.0,
                reading_score=0.0,
                objective_score=objectives_score,
                weighted_combined_score=0.0,
                difficulty_multiplier=1.0
            )

        weights = np.array(weights)
        weights = weights / np.sum(weights)

        combined_score = np.sum(np.array(scores) * weights)

        difficulty_multipliers = {
            'beginner': 0.9, 'easy': 0.9, 'medium': 1.0,
            'intermediate': 1.1, 'hard': 1.2, 'advanced': 1.25, 'expert': 1.3
        }

        raw_final_score = combined_score * difficulty_multipliers.get(difficulty_level.lower(), 1.0)
        clamped_raw = min(100.0, max(0.0, raw_final_score))

        # ✅ Normalize to 1–5 scale and round
        normalized_score = round(1 + (clamped_raw / 100) * 4, 2)

        return PADResult(
            final_score=round(clamped_raw, 2),         # Raw PAD score (0–100)
            normalized_final_score=normalized_score,   # Normalized score (1–5)
            transcript_score=np.mean(transcript_scores) if transcript_scores else 0.0,
            assignment_score=np.mean(assignment_scores) if assignment_scores else 0.0,
            reading_score=np.mean(reading_scores) if reading_scores else 0.0,
            objective_score=objectives_score,
            weighted_combined_score=round(combined_score, 2),
            difficulty_multiplier=difficulty_multipliers.get(difficulty_level.lower(), 1.0)
        )

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
        model_for_vision = genai.GenerativeModel('gemini-2.0-flash-001')
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

def read_reading(drive_file_id, service_account_file, custom_prompt):
    """Main function to download PDF from Drive temporarily and extract content with Gemini"""
    temp_file_path = None
    
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
        
        return {drive_file_id : extracted_content}
        
    except Exception as e:
        print(f"Error processing PDF from Drive: {str(e)}")
        return ""
        
    finally:
        # Always clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                print(f"Error deleting temporary file: {e}")
    
# Example usage
def example_dynamic_analysis():
    """Example of dynamic analysis that works for any course"""
    service_account_file = ""
    folder_id = ""
    service = get_gdrive_service(service_account_file)
    gemini_key = ""
    custom_extraction_prompt = None

    analyzer = DynamicModuleAnalyzer(gemini_api_key=gemini_key, service_account_file=service_account_file)
    calculator = MinimalPADCalculator(gemini_api_key=gemini_key, service_account_file=service_account_file)

    metadata = fetch_metadata_json(service, folder_id)
    course_info = metadata["About this Course"]
    difficulty_level = metadata["Level"].lower()
    extensions = ['.txt', '.pdf']
    all_files = find_files_recursively(service, folder_id, extensions)
    module_files = organize_files_by_module(metadata, all_files)

    module_results = []

    for key, value in metadata.items():
        if key.startswith("Module") and isinstance(value, dict):
            module_name = value.get("Name", key)
            print(f"Module --> {module_name}")
            learning_objectives = value.get("Learning Objectives", [])
            transcript_paths = [d["txt_file"] for d in module_files[module_name] if "txt_file" in d]
            reading_paths = [d["pdf_file"] for d in module_files[module_name] if "pdf_file" in d]
            assignment_paths = [d["odt_file"] for d in module_files[module_name] if "odt_file" in d]
            reading_contents = [
                read_reading(file_path, service_account_file, custom_extraction_prompt)
                for file_path in reading_paths
            ]

            module_input = ModuleInput(course_info, difficulty_level, learning_objectives, transcript_paths, reading_contents)

            result = analyzer.analyze_module_comprehensive(module_input)

            cccs_result = result["cccs_result"]
            tdb_result = result["tdb_result"]
            rwss_result = result["rwss_result"]

            # Normalize CCCS, TDB, RWSS
            normalized_cccs, normalized_tdb, normalized_rwss = analyzer.normalize_scores_to_1_5_scale(
                cccs_result.final_score,
                tdb_result.tdb_score,
                rwss_result.rwss_score
            )

            pad_result = calculator.calculate_pad_score(
                course_info, difficulty_level, learning_objectives,
                transcript_paths, assignment_paths, reading_contents
            )

            # Normalize PAD separately (since it's 0–100 → 1–5)
            normalized_pad_score = round(1 + (pad_result.final_score / 100) * 4, 2)

            # Generate final output JSON
            module_summary = generate_module_metric_summary(
                module_name=module_name,
                cccs_result=cccs_result,
                tdb_result=tdb_result,
                rwss_result=rwss_result,
                pad_result=pad_result,
                normalized_cccs=normalized_cccs,
                normalized_tdb=normalized_tdb,
                normalized_rwss=normalized_rwss,
                normalized_pad=normalized_pad_score
            )

            module_results.append(module_summary)

            break

    return module_results

def prompt_cccs_explanation(cccs_result: CCCSResult):
    learner_prompt = f"""
You are a course quality analyst evaluating educational content. The **Core Concept Coverage Score (CCCS)** reflects the extent to which essential ideas are introduced, developed, and reinforced in the module.

**Methodology & Directional Overview:**
- **Base Coverage (0 to 1):** Indicates the proportion of key concepts addressed out of the total expected. A higher value suggests broader concept inclusion.
- **Confidence Bonus (0 to 1):** Captures how confidently and unambiguously concepts are expressed. Direct and assertive phrasing strengthens this.
- **Semantic Coherence Bonus (0 to 1):** Measures how logically the concepts are linked, indicating integrated, structured knowledge rather than isolated facts.
- **Coverage Bonus (0 to 1):** Reflects evenness in how concepts are referenced. Balanced coverage across concepts earns higher values.
- **Objective Alignment Bonus (0 to 1):** Assesses how closely the covered concepts align with the module’s intended learning objectives.

**Final Score Derivation:**
The final CCCS score is a weighted aggregation of the above sub-metrics, normalized to a scale of **1 to 5**, where **1 indicates poor conceptual clarity and depth**, and **5 represents thorough, confident, and goal-aligned delivery**.

---
**Input:**
Metric: Core Concept Coverage Score (CCCS)  
Intermediate Metrics:
- Base Coverage: {cccs_result.base_coverage:.3f}
- Confidence Bonus: {cccs_result.confidence_bonus:.3f}
- Semantic Coherence Bonus: {cccs_result.semantic_coherence_bonus:.3f}
- Coverage Bonus: {cccs_result.coverage_bonus:.3f}
- Objective Alignment Bonus: {cccs_result.objective_alignment_bonus:.3f}
Final Rating (1–5): {cccs_result.final_score:.2f}

*Do not mention any numeric values or scores in the explanation.*
---
**Output Format (only this):**
<Explain to the learner how well this module covers essential concepts, how clearly these ideas are conveyed, and how aligned the material is to learning goals. Highlight conceptual clarity, completeness, integration, and identify any gaps or opportunities for conceptual strengthening. Do not mention any numeric scores.>
"""

    instructor_prompt = f"""
You are an instructional designer evaluating the conceptual design of your module. The **Core Concept Coverage Score (CCCS)** quantifies the extent, confidence, and alignment of your course’s conceptual content.

**Methodology & Directional Overview:**
- **Base Coverage (0 to 1):** Proportion of targeted key concepts actually covered.
- **Confidence Bonus (0 to 1):** Degree to which concepts are conveyed using clear, assertive language.
- **Semantic Coherence Bonus (0 to 1):** Evaluates the integration and logical progression of ideas.
- **Coverage Bonus (0 to 1):** Rewards consistent and distributed referencing of concepts throughout.
- **Objective Alignment Bonus (0 to 1):** Measures congruence between presented concepts and defined learning objectives.

**Final Score Derivation:**
The final CCCS is computed using a scaled combination of all these sub-metrics and presented on a **1–5 scale**, where **5 implies strong instructional design** and **1 reflects major conceptual gaps or poor structuring**.

---
**Input:**
Metric: Core Concept Coverage Score (CCCS)  
Intermediate Metrics:
- Base Coverage: {cccs_result.base_coverage:.3f}
- Confidence Bonus: {cccs_result.confidence_bonus:.3f}
- Semantic Coherence Bonus: {cccs_result.semantic_coherence_bonus:.3f}
- Coverage Bonus: {cccs_result.coverage_bonus:.3f}
- Objective Alignment Bonus: {cccs_result.objective_alignment_bonus:.3f}
Final Rating (1–5): {cccs_result.final_score:.2f}

*Do not mention any numeric values or scores in the explanation.*
---
**Output Format (only this):**
<Provide a focused evaluation of this module’s concept delivery. Validate strengths such as conceptual breadth, clarity, or alignment to learning objectives. Suggest any refinements in consistency, integration, or clarity without referencing scores.>
"""
    return learner_prompt, instructor_prompt


def prompt_tdb_explanation(tdb_result: TDBResult):
    learner_prompt = f"""
You are a student evaluating the depth and fairness of topic coverage in a module. The **Topic Distribution Balance (TDB)** metric shows how evenly learning time and focus are distributed across key topics.

**Methodology & Directional Overview:**
- **Content Distribution**: Evaluates the frequency of different topics across the module. Ideal modules distribute attention fairly without neglecting or overloading any single topic.
- **Cluster Analysis**: Identifies thematic clusters. High-quality modules demonstrate conceptual grouping, making learning feel structured and connected.
- **Imbalance Penalties**: Tracks overrepresented or underrepresented topics. A balanced module minimizes these penalties by ensuring all topics receive proportional focus.

**Final Score Derivation:**
The TDB score is computed by analyzing how fairly topics are covered, how cohesively they are grouped, and how much imbalance is present. Penalties are applied for skewed coverage, and the final score is scaled to a **1–5 range**, where **1 reflects highly uneven topic distribution**, and **5 indicates well-paced, balanced content**.

---
**Input:**
Metric: Topic Distribution Balance (TDB)  
Intermediate Metrics:
- Total Topics Detected: {len(tdb_result.content_distribution)}
- Topic Clusters Identified: {len(tdb_result.cluster_analysis)}
- Imbalance Penalties Count: {len(tdb_result.imbalance_penalties)}
- Total Penalty Weight: {sum(tdb_result.imbalance_penalties.values()):.3f}
Final Rating (1–5): {tdb_result.tdb_score:.2f}

*Do not mention any numeric values or scores in the explanation.*
---
**Output Format (only this):**
<Explain to the learner whether the module content feels well-balanced or overloaded. Help them understand which areas may feel repetitive, lacking, or too shallow based on topic distribution. Emphasize structural clarity, balance, and pacing. Do not mention any numeric values.>
"""

    instructor_prompt = f"""
You are an instructional strategist reviewing how evenly topics are addressed in a module. The **Topic Distribution Balance (TDB)** metric helps diagnose whether your course avoids topic overcrowding or neglect.

**Methodology & Directional Overview:**
- **Content Distribution**: Quantifies how proportionally each key topic is addressed throughout the module.
- **Cluster Analysis**: Detects conceptual groupings and checks whether topics are thematically coherent.
- **Imbalance Penalties**: Penalties are assigned when certain topics dominate or when expected topics are underrepresented or missing.

**Final Score Derivation:**
The TDB score is derived from topic frequency, quality of clustering, and penalty magnitude due to imbalance. This composite is scaled to a **1–5 score**, where **5 indicates optimal balance and instructional fairness**, and **1 indicates skewed focus or missing breadth**.

---
**Input:**
Metric: Topic Distribution Balance (TDB)  
Intermediate Metrics:
- Total Topics Detected: {len(tdb_result.content_distribution)}
- Topic Clusters Identified: {len(tdb_result.cluster_analysis)}
- Imbalance Penalties Count: {len(tdb_result.imbalance_penalties)}
- Total Penalty Weight: {sum(tdb_result.imbalance_penalties.values()):.3f}
Final Rating (1–5): {tdb_result.tdb_score:.2f}

*Do not mention any numeric values or scores in the explanation.*
---
**Output Format (only this):**
<Give a professional, structured evaluation of the module’s topic coverage. Comment on conceptual grouping, pacing, and whether any topics feel disproportionately covered or absent. Avoid numeric references and offer instructional suggestions.>
"""
    return learner_prompt, instructor_prompt

def prompt_rwss_explanation(rwss_result: RWSSResult):
    # Compute average industry coverage score
    industry_score = (
        sum(rwss_result.industry_coverage.values()) / len(rwss_result.industry_coverage)
        if rwss_result.industry_coverage else 0.0
    )

    learner_prompt = f"""
You are a learner trying to understand how relevant and applicable your course material is. The **Real-World Scenario Score (RWSS)** helps assess how well the module content connects theoretical concepts to professional, real-world contexts.

**Methodology & Directional Overview:**
- **Scenario Detection**: Tracks how many real-world situations, case studies, or domain examples are embedded. A higher count usually reflects richer contextualization.
- **Industry Coverage**: Measures the diversity of industries mentioned. The broader the range, the more adaptable and relatable the content becomes.
- **Tool Diversity**: Indicates whether the module references a variety of platforms, software tools, or frameworks common to professional practice.
- **Practical Application**: Assesses how well abstract theories are translated into real-life workflows or decision-making processes.

**Final Score Derivation:**
The RWSS score is computed from a weighted combination of these sub-metrics. Higher values across all indicate that the content is not only academic but also grounded in realistic practice. The final RWSS is scaled between **1 (low relevance)** and **5 (excellent real-world applicability)**.

**Valid Ranges for Intermediate Metrics:**
- **Detected Scenarios**: Typically ranges from **0 to 20+**, with higher counts indicating better contextual integration.
- **Industry Coverage Score**: Varies from **0.0 (no coverage)** to **1.0 (strong multi-domain representation)**.
- **Tool Diversity Score**: Between **0.0 (no tools)** to **1.0 (wide range of tools/platforms)**.
- **Practical Application Score**: From **0.0 (purely abstract)** to **1.0 (deeply practical and contextualized)**.

---
**Input:**
Metric: Real-World Scenario Score (RWSS)  
Intermediate Metrics:
- Detected Scenarios: {len(rwss_result.detected_scenarios)}
- Industry Coverage Score: {industry_score:.3f}
- Tool Diversity Score: {rwss_result.tool_diversity_score:.3f}
- Practical Application Score: {rwss_result.practical_application_score:.3f}
Final Rating (1–5): {rwss_result.rwss_score:.2f}

*Do not refer to numeric values directly in the output.*
---
**Output Format (only this):**
<Help the learner understand how relatable and useful the module content is for the real world. Highlight the strength of examples, relevance of industries and tools mentioned, and how clearly concepts connect to practical applications. Emphasize experiential value and suggest if content ever feels disconnected or abstract.>
"""

    instructor_prompt = f"""
You are an instructional designer evaluating the real-world relevance of your course. The **Real-World Scenario Score (RWSS)** reflects how effectively theoretical content is grounded in practical, industry-facing examples.

**Methodology & Directional Overview:**
- **Scenario Detection**: Detects the number of realistic scenarios or professional situations in the module. More scenarios indicate a richer real-world embedding.
- **Industry Coverage**: Evaluates the diversity of industries addressed. Broader coverage ensures learners see wider applicability.
- **Tool Diversity**: Assesses how many varied tools, platforms, or technologies are referenced, increasing practical preparedness.
- **Practical Application**: Determines how explicitly concepts are tied to actionable tasks or decisions relevant to the real world.

**Final Score Derivation:**
The final RWSS is calculated by integrating the above four metrics with weights emphasizing contextual applicability. Scores are normalized to a **1 to 5 range**, where **1 means weak real-world anchoring**, and **5 means high applied relevance**.

**Valid Ranges for Intermediate Metrics:**
- **Detected Scenarios**: 0 to 20+ typical, depending on course depth.
- **Industry Coverage Score**: 0.0 (narrow) to 1.0 (diverse multi-sector coverage).
- **Tool Diversity Score**: 0.0 (no tool references) to 1.0 (strong tool ecosystem representation).
- **Practical Application Score**: 0.0 (no clear application) to 1.0 (fully applied learning environment).

---
**Input:**
Metric: Real-World Scenario Score (RWSS)  
Intermediate Metrics:
- Detected Scenarios: {len(rwss_result.detected_scenarios)}
- Industry Coverage Score: {industry_score:.3f}
- Tool Diversity Score: {rwss_result.tool_diversity_score:.3f}
- Practical Application Score: {rwss_result.practical_application_score:.3f}
Final Rating (1–5): {rwss_result.rwss_score:.2f}

*Do not refer to numeric values directly in the output.*
---
**Output Format (only this):**
<Provide a detailed critique of how well the module integrates theory with practical reality. Mention the breadth of scenarios, the inclusion of domain-specific tools, and the diversity of industry representation. Suggest ways to enhance the practical utility of content, especially if examples feel sparse or overly abstract.>
"""
    return learner_prompt, instructor_prompt


def prompt_pad_explanation(pad_result: PADResult):
    learner_prompt = f"""
You are a learner trying to judge how practically useful the content in this module will be. The **Practical Application Density (PAD)** score reflects how frequently the material connects theory to practical examples, exercises, and real-world tasks.

**Methodology & Directional Overview:**
The PAD score is derived by assessing the presence and quality of applied learning signals across four core content types:
- **Transcript Score** (range: 0.0 to 1.0): Indicates the extent to which spoken lecture content demonstrates hands-on application or real-world scenarios.
- **Assignment Score** (range: 0.0 to 1.0): Captures how well assignments push learners to apply theory in realistic tasks.
- **Reading Score** (range: 0.0 to 1.0): Reflects the presence of case studies, how-to guides, or practical walk-throughs in course readings.
- **Objective Score** (range: 0.0 to 1.0): Measures how aligned the stated learning objectives are with actionable skills.

Each component is **weighted** based on content importance and combined into a **Weighted Combined Score (0.0 to 1.0)**. This is further adjusted using a **Difficulty Multiplier (typically 0.8 to 1.2)** to account for expected application depth at various difficulty levels (e.g., beginner vs. advanced).

**Final Rating Methodology:**  
The **PAD score ranges from 0 to 100**, where higher values indicate stronger emphasis on real-world relevance and applied learning.

---
**Input:**
Metric: Practical Application Density (PAD)  
Intermediate Metrics:
- Transcript Score: {pad_result.transcript_score:.3f}
- Assignment Score: {pad_result.assignment_score:.3f}
- Reading Score: {pad_result.reading_score:.3f}
- Objective Score: {pad_result.objective_score:.3f}
- Weighted Combined Score: {pad_result.weighted_combined_score:.3f}
- Difficulty Multiplier: {pad_result.difficulty_multiplier:.2f}
Final Rating(1-5) : {pad_result.normalized_final_score:.2f}
---
**Output Format (only this):**
<Explain to the learner how well this module emphasizes real-world learning. Highlight strong areas of practical relevance and note where more application or contextual examples could help.>
"""

    instructor_prompt = f"""
You are an instructional evaluator assessing the practical utility of your module. The **Practical Application Density (PAD)** score measures how effectively your course bridges theoretical content with real-world implementation.

**Methodology & Directional Overview:**
The PAD score is calculated using the following inputs:
- **Transcript Score** (0.0 to 1.0): Measures spoken content’s real-world contextualization.
- **Assignment Score** (0.0 to 1.0): Evaluates task alignment with professional or applied challenges.
- **Reading Score** (0.0 to 1.0): Checks reading content for walkthroughs, scenarios, or instructional cases.
- **Objective Score** (0.0 to 1.0): Judges learning objectives based on how clearly they define actionable competencies.

These are combined into a **Weighted Combined Score** (range: 0.0 to 1.0) according to relevance. The result is then scaled using a **Difficulty Multiplier** (e.g., beginner: ~0.8, advanced: ~1.2), accounting for expected application depth at different course levels.

**Final Rating Methodology:**  
The resulting PAD score ranges from **0 to 100**, where higher values signify stronger real-world anchoring. The score guides improvement by identifying underperforming content zones (e.g., weak assignments, abstract readings, generic objectives).

---
**Input:**
Metric: Practical Application Density (PAD)  
Intermediate Metrics:
- Transcript Score: {pad_result.transcript_score:.3f}
- Assignment Score: {pad_result.assignment_score:.3f}
- Reading Score: {pad_result.reading_score:.3f}
- Objective Score: {pad_result.objective_score:.3f}
- Weighted Combined Score: {pad_result.weighted_combined_score:.3f}
- Difficulty Multiplier: {pad_result.difficulty_multiplier:.2f}
Final Rating(1-5): {pad_result.normalized_final_score:.2f}
---
**Output Format (only this):**
<Assess the practical rigor of the module. Highlight high-utility sections and recommend areas that would benefit from more hands-on application or industry relevance.>
"""
    return learner_prompt, instructor_prompt


def generate_explanation(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No explanation generated."
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Explanation generation failed."


def generate_module_metric_summary(
    module_name: str,
    cccs_result: CCCSResult,
    tdb_result: TDBResult,
    rwss_result: RWSSResult,
    pad_result: PADResult,
    normalized_cccs: float,   # <-- Add normalized inputs
    normalized_tdb: float,
    normalized_rwss: float,
    normalized_pad: float
) -> dict:
    # CCCCS Explanations
    cccs_learner_prompt, cccs_instructor_prompt = prompt_cccs_explanation(cccs_result)
    cccs_learner = generate_explanation(cccs_learner_prompt)
    cccs_instructor = generate_explanation(cccs_instructor_prompt)

    # TDB Explanations
    tdb_learner_prompt, tdb_instructor_prompt = prompt_tdb_explanation(tdb_result)
    tdb_learner = generate_explanation(tdb_learner_prompt)
    tdb_instructor = generate_explanation(tdb_instructor_prompt)

    # RWSS Explanations
    rwss_learner_prompt, rwss_instructor_prompt = prompt_rwss_explanation(rwss_result)
    rwss_learner = generate_explanation(rwss_learner_prompt)
    rwss_instructor = generate_explanation(rwss_instructor_prompt)

    # PAD Explanations
    pad_learner_prompt, pad_instructor_prompt = prompt_pad_explanation(pad_result)
    pad_learner = generate_explanation(pad_learner_prompt)
    pad_instructor = generate_explanation(pad_instructor_prompt)

    # Return final structured summary
    return {
        "module_name": module_name,
        "metrics": {
            "Core Concept Coverage Score": {
                "score": round(normalized_cccs, 2),  # <-- Use normalized score
                "learner_explanation": cccs_learner,
                "instructor_explanation": cccs_instructor
            },
            "Topic Distribution Balance Score": {
                "score": round(normalized_tdb, 2),  # <-- Use normalized score
                "learner_explanation": tdb_learner,
                "instructor_explanation": tdb_instructor
            },
            "Real World Scenario Score": {
                "score": round(normalized_rwss, 2),  # <-- Use normalized score
                "learner_explanation": rwss_learner,
                "instructor_explanation": rwss_instructor
            },
            "Practical Application Density Score": {
                "score": round(normalized_pad, 2),  # If PAD is already normalized
                "learner_explanation": pad_learner,
                "instructor_explanation": pad_instructor
            }
        }
    }

if __name__ == "__main__":
    start_time = time.time()

    # Run full module analysis
    results = example_dynamic_analysis()

    # Optionally print each module's result in readable form
    print(json.dumps(results, indent=2))

    # Or save to a file
    with open("final_module_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal analysis time: {elapsed_minutes:.2f} minutes")
