"""
Image Text Translation Service for NurseConnect
Handles OCR text extraction and translation from medical images, documents, and photos
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import tempfile
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# OCR libraries
import pytesseract
import easyocr
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    language: str

@dataclass
class ImageTranslationResult:
    original_image_url: str
    extracted_text: str
    translated_text: str
    text_regions: List[TextRegion]
    original_language: str
    target_language: str
    confidence: float
    processing_time: float
    annotated_image_url: Optional[str] = None

class ImageTextTranslationService:
    def __init__(self):
        # Initialize OCR engines
        self.easyocr_reader = easyocr.Reader(['en', 'de', 'fr', 'it', 'es', 'pt', 'nl'])
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Configure Tesseract if available
        try:
            # Try to find tesseract executable
            pytesseract.pytesseract.tesseract_cmd = self._find_tesseract()
        except Exception as e:
            logger.warning(f"Tesseract not found: {e}")
    
    def _find_tesseract(self) -> str:
        """Find Tesseract executable path"""
        possible_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            'tesseract'  # System PATH
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or path == 'tesseract':
                return path
        
        raise FileNotFoundError("Tesseract executable not found")
    
    async def preprocess_image(self, image_data: bytes) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply image enhancement techniques
            
            # 1. Resize if too small
            height, width = cv_image.shape[:2]
            if width < 300 or height < 300:
                scale_factor = max(300/width, 300/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 2. Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 3. Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 4. Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 5. Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(processed)
            
            # Additional PIL enhancements
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(processed_image)
            processed_image = enhancer.enhance(2.0)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original image if preprocessing fails
            return Image.open(io.BytesIO(image_data))
    
    async def extract_text_tesseract(self, image: Image.Image, language: str = 'eng') -> List[TextRegion]:
        """Extract text using Tesseract OCR"""
        try:
            # Language mapping for Tesseract
            lang_map = {
                'en': 'eng',
                'de': 'deu',
                'fr': 'fra',
                'it': 'ita',
                'es': 'spa',
                'pt': 'por',
                'nl': 'nld'
            }
            
            tesseract_lang = lang_map.get(language, 'eng')
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?:;-()[]{}"\' '
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(
                image, 
                lang=tesseract_lang, 
                config=custom_config, 
                output_type=pytesseract.Output.DICT
            )
            
            text_regions = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 30:  # Confidence threshold
                    bbox = (
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    )
                    
                    text_regions.append(TextRegion(
                        text=text,
                        confidence=int(data['conf'][i]) / 100.0,
                        bbox=bbox,
                        language=language
                    ))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []
    
    async def extract_text_easyocr(self, image: Image.Image, languages: List[str] = ['en']) -> List[TextRegion]:
        """Extract text using EasyOCR"""
        try:
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(image_array, detail=1)
            
            text_regions = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Confidence threshold
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    width = int(max(x_coords) - min(x_coords))
                    height = int(max(y_coords) - min(y_coords))
                    
                    text_regions.append(TextRegion(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(x, y, width, height),
                        language=languages[0]  # EasyOCR doesn't specify detected language
                    ))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return []
    
    async def extract_text_paddle(self, image: Image.Image) -> List[TextRegion]:
        """Extract text using PaddleOCR"""
        try:
            # Convert PIL to numpy array
            image_array = np.array(image)
            
            # Extract text
            results = self.paddle_ocr.ocr(image_array, cls=True)
            
            text_regions = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox_points, (text, confidence) = line
                        
                        if confidence > 0.3:  # Confidence threshold
                            # Convert bbox format
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]
                            x, y = int(min(x_coords)), int(min(y_coords))
                            width = int(max(x_coords) - min(x_coords))
                            height = int(max(y_coords) - min(y_coords))
                            
                            text_regions.append(TextRegion(
                                text=text.strip(),
                                confidence=confidence,
                                bbox=(x, y, width, height),
                                language='en'  # PaddleOCR default
                            ))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return []
    
    async def extract_text_multi_engine(self, image: Image.Image, language: str = 'en') -> List[TextRegion]:
        """Extract text using multiple OCR engines and combine results"""
        try:
            all_regions = []
            
            # Try EasyOCR first (usually most accurate)
            easyocr_regions = await self.extract_text_easyocr(image, [language])
            all_regions.extend(easyocr_regions)
            
            # Try PaddleOCR
            paddle_regions = await self.extract_text_paddle(image)
            all_regions.extend(paddle_regions)
            
            # Try Tesseract if available
            try:
                tesseract_regions = await self.extract_text_tesseract(image, language)
                all_regions.extend(tesseract_regions)
            except Exception as e:
                logger.warning(f"Tesseract not available: {e}")
            
            # Remove duplicates and merge overlapping regions
            merged_regions = self._merge_text_regions(all_regions)
            
            return merged_regions
            
        except Exception as e:
            logger.error(f"Multi-engine OCR failed: {e}")
            return []
    
    def _merge_text_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge overlapping text regions and remove duplicates"""
        if not regions:
            return []
        
        # Sort by confidence (highest first)
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        merged = []
        for region in regions:
            # Check if this region overlaps significantly with any existing region
            is_duplicate = False
            for existing in merged:
                if self._regions_overlap(region.bbox, existing.bbox, threshold=0.7):
                    # If the new region has higher confidence, replace the existing one
                    if region.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(region)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(region)
        
        return merged
    
    def _regions_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # Calculate IoU (Intersection over Union)
        if union == 0:
            return False
        
        iou = intersection / union
        return iou > threshold
    
    async def create_annotated_image(self, original_image: Image.Image, text_regions: List[TextRegion]) -> str:
        """Create annotated image with text regions highlighted"""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            
            # Draw bounding boxes and text
            for region in text_regions:
                x, y, w, h = region.bbox
                
                # Draw rectangle
                color = (0, 255, 0) if region.confidence > 0.7 else (0, 255, 255)  # Green for high confidence, yellow for low
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence score
                label = f"{region.confidence:.2f}"
                cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Convert back to PIL and then to base64
            annotated_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Convert to base64
            buffer = io.BytesIO()
            annotated_pil.save(buffer, format='PNG')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Failed to create annotated image: {e}")
            return ""
    
    async def detect_language_from_text(self, text: str) -> Tuple[str, float]:
        """Detect language from extracted text"""
        try:
            # Use the existing language detection from main translation service
            from main import translation_service
            language, confidence, _ = await translation_service.detect_language(text)
            return language, confidence
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en", 0.5
    
    async def translate_image_text(
        self,
        image_data: bytes,
        target_language: str,
        source_language: Optional[str] = None,
        domain: str = "healthcare"
    ) -> ImageTranslationResult:
        """Complete image text translation pipeline"""
        start_time = datetime.now()
        
        try:
            # Convert image data to base64 for original image URL
            original_image_base64 = base64.b64encode(image_data).decode('utf-8')
            original_image_url = f"data:image/jpeg;base64,{original_image_base64}"
            
            # Preprocess image
            processed_image = await self.preprocess_image(image_data)
            
            # Extract text using multiple OCR engines
            text_regions = await self.extract_text_multi_engine(
                processed_image, 
                source_language or 'en'
            )
            
            if not text_regions:
                raise ValueError("No text detected in image")
            
            # Combine all extracted text
            extracted_text = " ".join([region.text for region in text_regions])
            
            # Detect language if not provided
            if not source_language:
                detected_lang, detection_confidence = await self.detect_language_from_text(extracted_text)
                source_language = detected_lang
                logger.info(f"Detected language: {source_language} (confidence: {detection_confidence})")
            
            # Translate the extracted text
            from main import translation_service
            translation_result = await translation_service.translate_text(
                text=extracted_text,
                source_lang=source_language,
                target_lang=target_language,
                domain=domain
            )
            
            # Create annotated image
            annotated_image_url = await self.create_annotated_image(
                Image.open(io.BytesIO(image_data)), 
                text_regions
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate overall confidence
            ocr_confidence = sum(region.confidence for region in text_regions) / len(text_regions)
            overall_confidence = (ocr_confidence + translation_result.confidence) / 2
            
            return ImageTranslationResult(
                original_image_url=original_image_url,
                extracted_text=extracted_text,
                translated_text=translation_result.translated_text,
                text_regions=text_regions,
                original_language=source_language,
                target_language=target_language,
                confidence=overall_confidence,
                processing_time=processing_time,
                annotated_image_url=annotated_image_url
            )
            
        except Exception as e:
            logger.error(f"Image text translation failed: {e}")
            raise
    
    async def extract_text_from_medical_document(
        self,
        image_data: bytes,
        document_type: str = "general"
    ) -> Dict[str, any]:
        """Extract structured text from medical documents"""
        try:
            # Preprocess image with medical document specific enhancements
            processed_image = await self.preprocess_image(image_data)
            
            # Extract text regions
            text_regions = await self.extract_text_multi_engine(processed_image)
            
            # Organize text by document type
            if document_type == "prescription":
                return await self._parse_prescription(text_regions)
            elif document_type == "lab_report":
                return await self._parse_lab_report(text_regions)
            elif document_type == "medical_record":
                return await self._parse_medical_record(text_regions)
            else:
                return {
                    "type": "general",
                    "text_regions": text_regions,
                    "full_text": " ".join([region.text for region in text_regions])
                }
                
        except Exception as e:
            logger.error(f"Medical document extraction failed: {e}")
            raise
    
    async def _parse_prescription(self, text_regions: List[TextRegion]) -> Dict[str, any]:
        """Parse prescription-specific information"""
        # Implementation for prescription parsing
        # This would include drug names, dosages, instructions, etc.
        return {
            "type": "prescription",
            "medications": [],
            "dosages": [],
            "instructions": [],
            "text_regions": text_regions
        }
    
    async def _parse_lab_report(self, text_regions: List[TextRegion]) -> Dict[str, any]:
        """Parse lab report specific information"""
        # Implementation for lab report parsing
        # This would include test names, values, reference ranges, etc.
        return {
            "type": "lab_report",
            "tests": [],
            "values": [],
            "reference_ranges": [],
            "text_regions": text_regions
        }
    
    async def _parse_medical_record(self, text_regions: List[TextRegion]) -> Dict[str, any]:
        """Parse medical record specific information"""
        # Implementation for medical record parsing
        # This would include patient info, diagnoses, treatments, etc.
        return {
            "type": "medical_record",
            "patient_info": {},
            "diagnoses": [],
            "treatments": [],
            "text_regions": text_regions
        }

# Global image translation service instance
image_translation_service = ImageTextTranslationService()