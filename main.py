"""
AI Translation Service for NurseConnect
Provides advanced translation capabilities with healthcare domain specialization
Enhanced with Voice and Image Translation features
"""

import os
import io
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    MarianMTModel,
    MarianTokenizer
)

import torch
from langdetect import detect, detect_langs
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import voice and image translation services
from services.voice_translation import voice_translation_service, VoiceTranslationResult
from services.image_translation import image_translation_service, ImageTranslationResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NurseConnect AI Translation Service",
    description="Advanced multilingual translation service with voice and image capabilities for healthcare professionals",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language code (ISO 639-1)")
    target_language: str = Field(..., description="Target language code (ISO 639-1)")
    domain: str = Field(default="general", description="Domain specialization (healthcare, general)")
    preserve_formatting: bool = Field(default=True, description="Preserve text formatting")

class VoiceTranslationRequest(BaseModel):
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
    domain: str = Field(default="healthcare", description="Domain specialization")
    generate_audio: bool = Field(default=True, description="Generate audio for translated text")

class ImageTranslationRequest(BaseModel):
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
    domain: str = Field(default="healthcare", description="Domain specialization")
    document_type: str = Field(default="general", description="Type of document (prescription, lab_report, medical_record, general)")

class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for language detection")

class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to translate")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    domain: str = Field(default="general", description="Domain specialization")

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    processing_time: float
    model_used: str

# class LanguageDetectionResponse(BaseModel):
#     language: str
#     confidence: float
#     all_languages: List[Dict[str, float]]

class LanguageConfidence(BaseModel):
    language: str
    confidence: float

class LanguageDetectionResponse(BaseModel):
    language: str
    confidence: float
    all_languages: List[LanguageConfidence]


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_processing_time: float

@dataclass
class TranslationModel:
    name: str
    tokenizer: any
    model: any
    supported_languages: List[str]
    domain: str

class TranslationService:
    def __init__(self):
        self.models: Dict[str, TranslationModel] = {}
        self.sentence_model = None
        self.language_detector = None
        self.healthcare_terms = self._load_healthcare_terms()
        self.translation_cache = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize only essential components"""
        logger.info("Initializing translation service...")
        
        try:
            # Load only lightweight components initially
            self.healthcare_terms = self._load_healthcare_terms()
            self.initialized = True
            logger.info("Translation service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize translation service: {e}")
            raise

    async def _load_model_if_needed(self, source_lang: str, target_lang: str, domain: str):
        """Load model only when needed"""
        model_key = f"{source_lang}-{target_lang}-{domain}"
        
        if model_key not in self.models:
            await self._load_specific_model(source_lang, target_lang, domain)

    async def _load_specific_model(self, source_lang: str, target_lang: str, domain: str):
        """Load a specific model on demand"""
        try:
            model_name = self._get_model_name(source_lang, target_lang, domain)
            
            if "mbart" in model_name:
                from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                model = MBartForConditionalGeneration.from_pretrained(model_name)
            else:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
            
            model_key = f"{source_lang}-{target_lang}-{domain}"
            self.models[model_key] = TranslationModel(
                name=model_name,
                tokenizer=tokenizer,
                model=model,
                supported_languages=[source_lang, target_lang],
                domain=domain
            )
            logger.info(f"Loaded model: {model_key}")
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")

    async def _load_healthcare_models(self):
        """Load healthcare-specialized translation models"""
        try:
            # Healthcare-specialized models (using domain-adapted models when available)
            healthcare_model_pairs = [
                ("en", "de", "Helsinki-NLP/opus-mt-en-de"),
                ("en", "fr", "Helsinki-NLP/opus-mt-en-fr"),
                ("en", "it", "Helsinki-NLP/opus-mt-en-it"),
                ("en", "es", "Helsinki-NLP/opus-mt-en-es"),
                ("de", "en", "Helsinki-NLP/opus-mt-de-en"),
                ("fr", "en", "Helsinki-NLP/opus-mt-fr-en"),
                ("it", "en", "Helsinki-NLP/opus-mt-it-en"),
                ("es", "en", "Helsinki-NLP/opus-mt-es-en"),
            ]
            
            for source, target, model_name in healthcare_model_pairs:
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    
                    key = f"{source}-{target}-healthcare"
                    self.models[key] = TranslationModel(
                        name=model_name,
                        tokenizer=tokenizer,
                        model=model,
                        supported_languages=[source, target],
                        domain="healthcare"
                    )
                    logger.info(f"Loaded healthcare model: {source} -> {target}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load healthcare model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading healthcare models: {e}")

    async def _load_general_models(self):
        """Load general purpose translation models"""
        try:
            # Multi-language model for broader coverage
            general_models = [
                ("facebook/mbart-large-50-many-to-many-mmt", "general"),
                ("Helsinki-NLP/opus-mt-mul-en", "multilingual-to-english"),
            ]
            
            for model_name, model_type in general_models:
                try:
                    if "mbart" in model_name:
                        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
                        model = MBartForConditionalGeneration.from_pretrained(model_name)
                    else:
                        tokenizer = MarianTokenizer.from_pretrained(model_name)
                        model = MarianMTModel.from_pretrained(model_name)
                    
                    self.models[f"general-{model_type}"] = TranslationModel(
                        name=model_name,
                        tokenizer=tokenizer,
                        model=model,
                        supported_languages=["multi"],
                        domain="general"
                    )
                    logger.info(f"Loaded general model: {model_type}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load general model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading general models: {e}")

    def _load_healthcare_terms(self) -> Dict[str, Dict[str, str]]:
        """Load healthcare terminology dictionaries"""
        return {
            "en": {
                "dementia": "dementia",
                "alzheimer": "Alzheimer's disease",
                "palliative": "palliative care",
                "rehabilitation": "rehabilitation",
                "wound care": "wound care",
                "medication": "medication",
                "nursing": "nursing",
                "long-term care": "long-term care",
                "resident": "resident",
                "patient": "patient"
            },
            "de": {
                "dementia": "Demenz",
                "alzheimer": "Alzheimer-Krankheit",
                "palliative": "Palliativpflege",
                "rehabilitation": "Rehabilitation",
                "wound care": "Wundversorgung",
                "medication": "Medikation",
                "nursing": "Krankenpflege",
                "long-term care": "Langzeitpflege",
                "resident": "Bewohner",
                "patient": "Patient"
            },
            "fr": {
                "dementia": "démence",
                "alzheimer": "maladie d'Alzheimer",
                "palliative": "soins palliatifs",
                "rehabilitation": "réhabilitation",
                "wound care": "soins des plaies",
                "medication": "médication",
                "nursing": "soins infirmiers",
                "long-term care": "soins de longue durée",
                "resident": "résident",
                "patient": "patient"
            },
            "it": {
                "dementia": "demenza",
                "alzheimer": "malattia di Alzheimer",
                "palliative": "cure palliative",
                "rehabilitation": "riabilitazione",
                "wound care": "cura delle ferite",
                "medication": "farmaci",
                "nursing": "assistenza infermieristica",
                "long-term care": "assistenza a lungo termine",
                "resident": "residente",
                "patient": "paziente"
            }
        }

    async def detect_language(self, text: str) -> Tuple[str, float, List[Dict[str, float]]]:
        """Detect language of input text with confidence scores"""
        try:
            # Use langdetect for primary detection
            detected_langs = detect_langs(text)
            primary_lang = detected_langs[0].lang
            primary_confidence = detected_langs[0].prob
            
            # Format all detected languages
            all_languages = [
                {"language": lang.lang, "confidence": lang.prob}
                for lang in detected_langs
            ]
            
            return primary_lang, primary_confidence, all_languages
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en", 0.5, [{"language": "en", "confidence": 0.5}]

    def _select_best_model(self, source_lang: str, target_lang: str, domain: str) -> Optional[TranslationModel]:
        """Select the best available model for the language pair and domain"""
        
        # First try healthcare-specific model
        if domain == "healthcare":
            healthcare_key = f"{source_lang}-{target_lang}-healthcare"
            if healthcare_key in self.models:
                return self.models[healthcare_key]
        
        # Try general models
        general_key = f"{source_lang}-{target_lang}-general"
        if general_key in self.models:
            return self.models[general_key]
        
        # Fall back to multilingual models
        for key, model in self.models.items():
            if "general" in key and "multi" in model.supported_languages:
                return model
        
        return None

    def _preprocess_text(self, text: str, preserve_formatting: bool = True) -> str:
        """Preprocess text before translation"""
        if not preserve_formatting:
            # Basic cleaning
            text = text.strip()
            # Remove extra whitespace
            text = " ".join(text.split())
        
        return text

    def _postprocess_translation(self, translated_text: str, original_text: str, 
                                target_lang: str, domain: str) -> str:
        """Post-process translation with domain-specific improvements"""
        
        if domain == "healthcare":
            # Apply healthcare terminology corrections
            translated_text = self._apply_healthcare_terminology(translated_text, target_lang)
        
        # Preserve original formatting patterns
        translated_text = self._preserve_formatting(translated_text, original_text)
        
        return translated_text

    def _apply_healthcare_terminology(self, text: str, target_lang: str) -> str:
        """Apply healthcare-specific terminology corrections"""
        if target_lang not in self.healthcare_terms:
            return text
        
        terms = self.healthcare_terms[target_lang]
        
        # Apply term replacements (case-insensitive)
        for en_term, target_term in terms.items():
            # Simple replacement - in production, use more sophisticated NER
            text = text.replace(en_term.lower(), target_term.lower())
            text = text.replace(en_term.title(), target_term.title())
            text = text.replace(en_term.upper(), target_term.upper())
        
        return text

    def _preserve_formatting(self, translated_text: str, original_text: str) -> str:
        """Preserve formatting from original text"""
        # Basic formatting preservation
        if original_text.endswith('...'):
            if not translated_text.endswith('...'):
                translated_text += '...'
        
        if original_text.endswith('!'):
            if not translated_text.endswith('!'):
                translated_text += '!'
        
        return translated_text

    def _calculate_translation_confidence(self, original: str, translated: str, 
                                        source_lang: str, target_lang: str) -> float:
        """Calculate translation confidence using semantic similarity"""
        try:
            if self.sentence_model is None:
                return 0.8  # Default confidence
            
            # Get embeddings for both texts
            original_embedding = self.sentence_model.encode([original])
            translated_embedding = self.sentence_model.encode([translated])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(original_embedding, translated_embedding)[0][0]
            
            # Adjust confidence based on language pair difficulty
            language_difficulty_factor = self._get_language_difficulty_factor(source_lang, target_lang)
            
            confidence = similarity * language_difficulty_factor
            return min(max(confidence, 0.1), 0.99)  # Clamp between 0.1 and 0.99
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.75  # Default confidence

    def _get_language_difficulty_factor(self, source_lang: str, target_lang: str) -> float:
        """Get difficulty factor based on language pair"""
        # Language family similarity factors
        romance_languages = {"es", "fr", "it", "pt", "ro"}
        germanic_languages = {"en", "de", "nl", "sv", "no", "da"}
        
        if source_lang in romance_languages and target_lang in romance_languages:
            return 0.95
        elif source_lang in germanic_languages and target_lang in germanic_languages:
            return 0.90
        elif (source_lang in romance_languages and target_lang in germanic_languages) or \
             (source_lang in germanic_languages and target_lang in romance_languages):
            return 0.85
        else:
            return 0.80

    async def translate_text(self, text: str, source_lang: str, target_lang: str, 
                           domain: str = "general", preserve_formatting: bool = True) -> TranslationResponse:
        """Translate text using the best available model"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{hash(text)}_{source_lang}_{target_lang}_{domain}"
            if cache_key in self.translation_cache:
                cached_result = self.translation_cache[cache_key]
                cached_result.processing_time = (datetime.now() - start_time).total_seconds()
                return cached_result
            
            # Preprocess text
            processed_text = self._preprocess_text(text, preserve_formatting)
            
            # Select best model
            model = self._select_best_model(source_lang, target_lang, domain)
            if not model:
                raise HTTPException(status_code=400, 
                                  detail=f"No model available for {source_lang} -> {target_lang}")
            
            # Perform translation
            if "mbart" in model.name:
                translated_text = await self._translate_with_mbart(processed_text, source_lang, 
                                                                 target_lang, model)
            else:
                translated_text = await self._translate_with_marian(processed_text, model)
            
            # Post-process translation
            final_translation = self._postprocess_translation(translated_text, text, 
                                                             target_lang, domain)
            
            # Calculate confidence
            confidence = self._calculate_translation_confidence(text, final_translation, 
                                                              source_lang, target_lang)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TranslationResponse(
                translated_text=final_translation,
                source_language=source_lang,
                target_language=target_lang,
                confidence=confidence,
                processing_time=processing_time,
                model_used=model.name
            )
            
            # Cache result
            self.translation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    async def _translate_with_marian(self, text: str, model: TranslationModel) -> str:
        """Translate using Marian model"""
        try:
            inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.model.generate(**inputs, max_length=512, num_beams=4, 
                                             early_stopping=True)
            
            translated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text
            
        except Exception as e:
            logger.error(f"Marian translation failed: {e}")
            raise

    async def _translate_with_mbart(self, text: str, source_lang: str, 
                                  target_lang: str, model: TranslationModel) -> str:
        """Translate using mBART model"""
        try:
            # mBART language code mapping
            lang_code_map = {
                "en": "en_XX", "de": "de_DE", "fr": "fr_XX", "it": "it_IT", "es": "es_XX", "pt": "pt_XX",
    "nl": "nl_XX", "sv": "sv_SE", "no": "no_XX", "da": "da_DK", "fi": "fi_FI", "pl": "pl_PL",
    "hi": "hi_IN", "ar": "ar_AR", "ru": "ru_RU", "zh": "zh_CN", "ja": "ja_XX", "ko": "ko_KR",
    "tr": "tr_TR", "th": "th_TH", "vi": "vi_VN", "uk": "uk_UA"
            }
            
            src_lang_code = lang_code_map.get(source_lang, "en_XX")
            tgt_lang_code = lang_code_map.get(target_lang, "en_XX")
            
            model.tokenizer.src_lang = src_lang_code
            inputs = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.model.generate(**inputs, 
                                             forced_bos_token_id=model.tokenizer.lang_code_to_id[tgt_lang_code],
                                             max_length=512, num_beams=4, early_stopping=True)
            
            translated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text
            
        except Exception as e:
            logger.error(f"mBART translation failed: {e}")
            raise

# Initialize translation service
translation_service = TranslationService()

@app.on_event("startup")
async def startup_event():
    """Initialize the translation service on startup"""
    await translation_service.initialize()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "NurseConnect AI Translation Service",
        "status": "healthy",
        "version": "2.0.0",
        "features": ["text", "voice", "image"],
        "available_models": len(translation_service.models),
        "supported_domains": ["general", "healthcare"]
    }

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """Detect the language of input text"""
    try:
        language, confidence, all_languages = await translation_service.detect_language(request.text)
        
        return LanguageDetectionResponse(
            language=language,
            confidence=confidence,
            all_languages=all_languages
        )
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate text from source to target language"""
    try:
        result = await translation_service.translate_text(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
            domain=request.domain,
            preserve_formatting=request.preserve_formatting
        )
        return result
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate-batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """Translate multiple texts in batch"""
    start_time = datetime.now()
    
    try:
        translations = []
        
        for text in request.texts:
            result = await translation_service.translate_text(
                text=text,
                source_lang=request.source_language,
                target_lang=request.target_language,
                domain=request.domain
            )
            translations.append(result)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        return BatchTranslationResponse(
            translations=translations,
            total_processing_time=total_time
        )
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

@app.post("/translate-voice")
async def translate_voice(
    audio_file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None),
    domain: str = Form("healthcare"),
    generate_audio: bool = Form(True)
):
    """Translate voice/audio content"""
    try:
        # Read audio file
        audio_data = await audio_file.read()
        print(f"Received audio length: {len(audio_data)} bytes, filename: {audio_file.filename}")

        # ✅ Convert to WAV if not already in WAV format
        if not audio_file.filename.lower().endswith(".wav"):
            print("Converting audio to WAV format...")
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz for STT
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            audio_data = wav_io.getvalue()

        # Process voice translation
        result = await voice_translation_service.translate_voice(
            audio_data=audio_data,
            target_language=target_language,
            source_language=source_language,
            domain=domain,
            generate_audio=generate_audio
        )

        return {
            "original_text": result.original_text,
            "translated_text": result.translated_text,
            "original_language": result.original_language,
            "target_language": result.target_language,
            "confidence": result.confidence,
            "audio_duration": result.audio_duration,
            "processing_time": result.processing_time,
            "audio_url": result.audio_url
        }

    except Exception as e:
        logger.exception("Voice translation error (full traceback):")
        raise HTTPException(status_code=500, detail=f"Voice translation failed: {str(e)}")
       

@app.post("/translate-image")
async def translate_image(
    image_file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: Optional[str] = Form(None),
    domain: str = Form("healthcare"),
    document_type: str = Form("general")
):
    """Translate text from images"""
    try:
        # Read image file
        image_data = await image_file.read()
        
        # Process image translation
        result = await image_translation_service.translate_image_text(
            image_data=image_data,
            target_language=target_language,
            source_language=source_language,
            domain=domain
        )
        
        return {
            "original_image_url": result.original_image_url,
            "extracted_text": result.extracted_text,
            "translated_text": result.translated_text,
            "text_regions": [
                {
                    "text": region.text,
                    "confidence": region.confidence,
                    "bbox": region.bbox,
                    "language": region.language
                }
                for region in result.text_regions
            ],
            "original_language": result.original_language,
            "target_language": result.target_language,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "annotated_image_url": result.annotated_image_url
        }
        
    except Exception as e:
        logger.error(f"Image translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Image translation failed: {str(e)}")

@app.post("/extract-medical-document")
async def extract_medical_document(
    image_file: UploadFile = File(...),
    document_type: str = Form("general")
):
    """Extract structured text from medical documents"""
    try:
        # Read image file
        image_data = await image_file.read()
        
        # Extract structured text
        result = await image_translation_service.extract_text_from_medical_document(
            image_data=image_data,
            document_type=document_type
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Medical document extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Document extraction failed: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get information about available translation models"""
    models_info = {}
    for key, model in translation_service.models.items():
        models_info[key] = {
            "name": model.name,
            "supported_languages": model.supported_languages,
            "domain": model.domain
        }
    return models_info

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(translation_service.models),
        "cache_size": len(translation_service.translation_cache),
        "sentence_model_loaded": translation_service.sentence_model is not None,
        "language_detector_loaded": translation_service.language_detector is not None,
        "voice_service_available": voice_translation_service is not None,
        "image_service_available": image_translation_service is not None,
        "features": ["text_translation", "voice_translation", "image_translation", "medical_document_extraction"]
    }
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)