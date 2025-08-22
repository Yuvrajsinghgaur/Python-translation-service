"""
AI Translation Service for NurseConnect
Provides advanced translation capabilities with healthcare domain specialization
Enhanced with Voice Translation feature only (image translation code commented out)
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
from langdetect import detect, detect_langs
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Import only voice translation service, image translation import commented out
from services.voice_translation import voice_translation_service, VoiceTranslationResult
# from services.image_translation import image_translation_service, ImageTranslationResult  # Commented out



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


# Commented out image translation models
# class ImageTranslationRequest(BaseModel):
#     target_language: str = Field(..., description="Target language code")
#     source_language: Optional[str] = Field(None, description="Source language code (auto-detect if not provided)")
#     domain: str = Field(default="healthcare", description="Domain specialization")
#     document_type: str = Field(default="general", description="Type of document (prescription, lab_report, medical_record, general)")


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


    # ... (rest of the TranslationService class code unchanged) ...


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
        "features": ["text", "voice"],
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


        # Convert to WAV if not already in WAV format
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


# Image translation endpoints and imports commented out as not currently used
# @app.post("/translate-image")
# async def translate_image(
#     image_file: UploadFile = File(...),
#     target_language: str = Form(...),
#     source_language: Optional[str] = Form(None),
#     domain: str = Form("healthcare"),
#     document_type: str = Form("general")
# ):
#     """Translate text from images"""
#     try:
#         image_data = await image_file.read()
#         result = await image_translation_service.translate_image_text(
#             image_data=image_data,
#             target_language=target_language,
#             source_language=source_language,
#             domain=domain
#         )
#         return {
#             "original_image_url": result.original_image_url,
#             "extracted_text": result.extracted_text,
#             "translated_text": result.translated_text,
#             "text_regions": [
#                 {
#                     "text": region.text,
#                     "confidence": region.confidence,
#                     "bbox": region.bbox,
#                     "language": region.language
#                 }
#                 for region in result.text_regions
#             ],
#             "original_language": result.original_language,
#             "target_language": result.target_language,
#             "confidence": result.confidence,
#             "processing_time": result.processing_time,
#             "annotated_image_url": result.annotated_image_url
#         }
#     except Exception as e:
#         logger.error(f"Image translation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Image translation failed: {str(e)}")


# @app.post("/extract-medical-document")
# async def extract_medical_document(
#     image_file: UploadFile = File(...),
#     document_type: str = Form("general")
# ):
#     """Extract structured text from medical documents"""
#     try:
#         image_data = await image_file.read()
#         result = await image_translation_service.extract_text_from_medical_document(
#             image_data=image_data,
#             document_type=document_type
#         )
#         return result
#     except Exception as e:
#         logger.error(f"Medical document extraction error: {e}")
#         raise HTTPException(status_code=500, detail=f"Document extraction failed: {str(e)}")


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
        "image_service_available": False,  # image translation disabled
        "features": ["text_translation", "voice_translation"]
    }

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Render provides PORT environment variable
#     uvicorn.run(app, host="0.0.0.0", port=port)
