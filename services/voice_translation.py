# """
# Voice Translation Service for NurseConnect
# Handles speech-to-text, translation, and text-to-speech for multilingual communication
# """

# import os
# import asyncio
# import logging
# from typing import Dict, List, Optional, Tuple, BinaryIO
# from dataclasses import dataclass
# from datetime import datetime
# import tempfile
# import io
# import base64

# import sounddevice as sd
# import soundfile as sf
# import numpy as np
# import speech_recognition as sr
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# from gtts import gTTS

# from googletrans import Translator

# translator = Translator()


# logger = logging.getLogger(__name__)

# @dataclass
# class VoiceTranslationResult:
#     original_text: str
#     translated_text: str
#     original_language: str
#     target_language: str
#     confidence: float
#     audio_duration: float
#     processing_time: float
#     audio_url: Optional[str] = None


# class VoiceTranslationService:
#     def __init__(self):
#         self.recognizer = sr.Recognizer()
#         # ❌ Removed PyAudio Microphone dependency

#     async def record_audio(self, duration: int = 5, samplerate: int = 16000) -> bytes:
#         """
#         Record audio from microphone using sounddevice and return WAV bytes.
#         """
#         try:
#             logger.info(f"Recording {duration}s of audio at {samplerate}Hz...")
#             recording = sd.rec(
#                 int(duration * samplerate),
#                 samplerate=samplerate,
#                 channels=1,
#                 dtype="int16"
#             )
#             sd.wait()
#             with io.BytesIO() as buffer:
#                 sf.write(buffer, recording, samplerate, format="WAV")
#                 return buffer.getvalue()
#         except Exception as e:
#             logger.error(f"Microphone recording failed: {e}")
#             return b""

#     async def transcribe_audio(self, audio_data: bytes, language: str = 'en') -> Tuple[str, float]:
#         """Transcribe audio to text using speech recognition"""
#         temp_file_path = None
#         try:
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
#                 temp_file.write(audio_data)
#                 temp_file_path = temp_file.name

#             if not os.path.exists(temp_file_path):
#                 raise IOError("Temporary audio file was not created successfully")

#             with sr.AudioFile(temp_file_path) as source:
#                 audio = self.recognizer.record(source)

#             try:
#                 text = self.recognizer.recognize_google(audio, language=language)
#                 confidence = 0.85  # Google SR does not provide confidence
#                 return text, confidence
#             except sr.UnknownValueError:
#                 logger.warning("Could not understand audio")
#                 return "", 0.0
#             except sr.RequestError as e:
#                 logger.error(f"Speech recognition error: {e}")
#                 return "", 0.0
#         except Exception as e:
#             logger.error(f"Audio transcription failed: {e}")
#             return "", 0.0
#         finally:
#             if temp_file_path and os.path.exists(temp_file_path):
#                 try:
#                     os.unlink(temp_file_path)
#                 except Exception as e:
#                     logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

#     async def enhance_audio_quality(self, audio_data: bytes) -> bytes:
#         """Enhance audio quality for better transcription"""
#         try:
#             with io.BytesIO(audio_data) as audio_buffer:
#                 audio = AudioSegment.from_file(audio_buffer)

#             normalized_audio = audio.normalize()
#             chunks = split_on_silence(
#                 normalized_audio,
#                 min_silence_len=500,
#                 silence_thresh=normalized_audio.dBFS - 14,
#                 keep_silence=250
#             )

#             enhanced_audio = AudioSegment.empty()
#             for chunk in chunks:
#                 enhanced_audio += chunk

#             with io.BytesIO() as output_buffer:
#                 enhanced_audio.export(output_buffer, format="wav")
#                 return output_buffer.getvalue()
#         except Exception as e:
#             logger.warning(f"Audio enhancement failed, using original: {e}")
#             return audio_data

#     async def process_audio_file(self, file: BinaryIO) -> bytes:
#         """Process uploaded audio file to standard format"""
#         try:
#             file_content = file.read()
#             audio = AudioSegment.from_file(io.BytesIO(file_content))
#             audio = audio.set_frame_rate(16000).set_channels(1)
#             with io.BytesIO() as output_buffer:
#                 audio.export(output_buffer, format="wav")
#                 return output_buffer.getvalue()
#         except Exception as e:
#             logger.error(f"Audio file processing failed: {e}")
#             raise ValueError("Could not process audio file")

#     async def generate_speech(self, text: str, language: str) -> bytes:
#         """Generate speech audio from translated text"""
#         try:
#             tts = gTTS(text=text, lang=language)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
#                 tts.save(tmp_file.name)
#                 tmp_file.seek(0)
#                 audio_bytes = tmp_file.read()
#             os.unlink(tmp_file.name)
#             return audio_bytes
#         except Exception as e:
#             logger.error(f"Failed to generate speech: {e}")
#             return b""

#     async def detect_language_from_audio(self, audio_data: bytes) -> Tuple[str, float]:
#         """Dummy language detection (replace with actual model if needed)"""
#         return "en", 1.0  # fallback

# async def translate_voice(
#     self,
#     audio_data: bytes,
#     target_language: str,
#     source_language: Optional[str] = None,
#     domain: str = "healthcare",
#     generate_audio: bool = True
# ) -> VoiceTranslationResult:
#     start_time = datetime.now()

#     try:
#         processed_audio = await self.process_audio_file(io.BytesIO(audio_data))
#         enhanced_audio = await self.enhance_audio_quality(processed_audio)

#         with io.BytesIO(enhanced_audio) as audio_buffer:
#             audio_segment = AudioSegment.from_file(audio_buffer)
#             audio_duration = len(audio_segment) / 1000.0

#         if not source_language:
#             source_language, _ = await self.detect_language_from_audio(enhanced_audio)

#         original_text, confidence = await self.transcribe_audio(enhanced_audio, source_language)

#         if not original_text.strip():
#             raise ValueError("No speech detected in audio")

#         # Use googletrans Translator directly here:
#         try:
#             translation_result = translator.translate(original_text, src=source_language, dest=target_language)
#             translated_text = translation_result.text
#             translation_confidence = 0.9  # No confidence from googletrans, estimate high
#         except Exception as e:
#             raise ValueError(f"Google translation failed: {str(e)}")

#         audio_url = None
#         if generate_audio and translated_text:
#             translated_audio = await self.generate_speech(
#                 translated_text,
#                 target_language
#             )
#             if translated_audio:
#                 audio_base64 = base64.b64encode(translated_audio).decode('utf-8')
#                 audio_url = f"data:audio/mp3;base64,{audio_base64}"

#         processing_time = (datetime.now() - start_time).total_seconds()

#         return VoiceTranslationResult(
#             original_text=original_text,
#             translated_text=translated_text,
#             original_language=source_language,
#             target_language=target_language,
#             confidence=confidence * translation_confidence,
#             audio_duration=audio_duration,
#             processing_time=processing_time,
#             audio_url=audio_url
#         )

#     except Exception as e:
#         logger.error(f"Voice translation failed: {str(e)}", exc_info=True)
#         raise ValueError(f"Voice translation failed: {str(e)}")



# # Global voice translation service instance
# voice_translation_service = VoiceTranslationService()

"""
Voice Translation Service for NurseConnect
Handles speech-to-text, translation, and text-to-speech for multilingual communication
"""


import os
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import tempfile
import io
import base64


import soundfile as sf
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS


from googletrans import Translator


translator = Translator()


logger = logging.getLogger(__name__)


@dataclass
class VoiceTranslationResult:
    original_text: str
    translated_text: str
    original_language: str
    target_language: str
    confidence: float
    audio_duration: float
    processing_time: float
    audio_url: Optional[str] = None



class VoiceTranslationService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # ❌ Removed PyAudio Microphone and sounddevice dependency


    async def transcribe_audio(self, audio_data: bytes, language: str = 'en') -> Tuple[str, float]:
        """Transcribe audio to text using speech recognition"""
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name


            if not os.path.exists(temp_file_path):
                raise IOError("Temporary audio file was not created successfully")


            with sr.AudioFile(temp_file_path) as source:
                audio = self.recognizer.record(source)


            try:
                text = self.recognizer.recognize_google(audio, language=language)
                confidence = 0.85  # Google SR does not provide confidence
                return text, confidence
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return "", 0.0
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return "", 0.0
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return "", 0.0
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_file_path}: {e}")


    async def enhance_audio_quality(self, audio_data: bytes) -> bytes:
        """Enhance audio quality for better transcription"""
        try:
            with io.BytesIO(audio_data) as audio_buffer:
                audio = AudioSegment.from_file(audio_buffer)


            normalized_audio = audio.normalize()
            chunks = split_on_silence(
                normalized_audio,
                min_silence_len=500,
                silence_thresh=normalized_audio.dBFS - 14,
                keep_silence=250
            )


            enhanced_audio = AudioSegment.empty()
            for chunk in chunks:
                enhanced_audio += chunk


            with io.BytesIO() as output_buffer:
                enhanced_audio.export(output_buffer, format="wav")
                return output_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Audio enhancement failed, using original: {e}")
            return audio_data


    async def process_audio_file(self, file: BinaryIO) -> bytes:
        """Process uploaded audio file to standard format"""
        try:
            file_content = file.read()
            audio = AudioSegment.from_file(io.BytesIO(file_content))
            audio = audio.set_frame_rate(16000).set_channels(1)
            with io.BytesIO() as output_buffer:
                audio.export(output_buffer, format="wav")
                return output_buffer.getvalue()
        except Exception as e:
            logger.error(f"Audio file processing failed: {e}")
            raise ValueError("Could not process audio file")


    async def generate_speech(self, text: str, language: str) -> bytes:
        """Generate speech audio from translated text"""
        try:
            tts = gTTS(text=text, lang=language)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                tmp_file.seek(0)
                audio_bytes = tmp_file.read()
            os.unlink(tmp_file.name)
            return audio_bytes
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            return b""


    async def detect_language_from_audio(self, audio_data: bytes) -> Tuple[str, float]:
        """Dummy language detection (replace with actual model if needed)"""
        return "en", 1.0  # fallback


    async def translate_voice(
        self,
        audio_data: bytes,
        target_language: str,
        source_language: Optional[str] = None,
        domain: str = "healthcare",
        generate_audio: bool = True
    ) -> VoiceTranslationResult:
        start_time = datetime.now()


        try:
            processed_audio = await self.process_audio_file(io.BytesIO(audio_data))
            enhanced_audio = await self.enhance_audio_quality(processed_audio)


            with io.BytesIO(enhanced_audio) as audio_buffer:
                audio_segment = AudioSegment.from_file(audio_buffer)
                audio_duration = len(audio_segment) / 1000.0


            if not source_language:
                source_language, _ = await self.detect_language_from_audio(enhanced_audio)


            original_text, confidence = await self.transcribe_audio(enhanced_audio, source_language)


            if not original_text.strip():
                raise ValueError("No speech detected in audio")


            # Use googletrans Translator directly here:
            try:
                translation_result = translator.translate(original_text, src=source_language, dest=target_language)
                translated_text = translation_result.text
                translation_confidence = 0.9  # No confidence from googletrans, estimate high
            except Exception as e:
                raise ValueError(f"Google translation failed: {str(e)}")


            audio_url = None
            if generate_audio and translated_text:
                translated_audio = await self.generate_speech(
                    translated_text,
                    target_language
                )
                if translated_audio:
                    audio_base64 = base64.b64encode(translated_audio).decode('utf-8')
                    audio_url = f"data:audio/mp3;base64,{audio_base64}"


            processing_time = (datetime.now() - start_time).total_seconds()


            return VoiceTranslationResult(
                original_text=original_text,
                translated_text=translated_text,
                original_language=source_language,
                target_language=target_language,
                confidence=confidence * translation_confidence,
                audio_duration=audio_duration,
                processing_time=processing_time,
                audio_url=audio_url
            )


        except Exception as e:
            logger.error(f"Voice translation failed: {str(e)}", exc_info=True)
            raise ValueError(f"Voice translation failed: {str(e)}")




# Global voice translation service instance
voice_translation_service = VoiceTranslationService()

