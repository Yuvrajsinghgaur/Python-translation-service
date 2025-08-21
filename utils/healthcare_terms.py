"""
Healthcare terminology and domain-specific processing utilities
"""

from typing import Dict, List, Set
import json
import re

class HealthcareTerminologyProcessor:
    def __init__(self):
        self.terminology_db = self._load_terminology_database()
        self.medical_abbreviations = self._load_medical_abbreviations()
        self.drug_names = self._load_drug_names()
        
    def _load_terminology_database(self) -> Dict[str, Dict[str, str]]:
        """Load comprehensive healthcare terminology database"""
        return {
            "en": {
                # Conditions and Diseases
                "alzheimer": "Alzheimer's disease",
                "dementia": "dementia",
                "parkinson": "Parkinson's disease",
                "diabetes": "diabetes",
                "hypertension": "hypertension",
                "copd": "chronic obstructive pulmonary disease",
                "pneumonia": "pneumonia",
                "uti": "urinary tract infection",
                "dvt": "deep vein thrombosis",
                "stroke": "cerebrovascular accident",
                
                # Care Types
                "palliative": "palliative care",
                "hospice": "hospice care",
                "rehabilitation": "rehabilitation",
                "physical therapy": "physical therapy",
                "occupational therapy": "occupational therapy",
                "speech therapy": "speech therapy",
                
                # Medical Procedures
                "wound care": "wound care",
                "medication administration": "medication administration",
                "vital signs": "vital signs monitoring",
                "blood pressure": "blood pressure",
                "glucose monitoring": "glucose monitoring",
                "catheter care": "catheter care",
                
                # Healthcare Roles
                "registered nurse": "registered nurse",
                "licensed practical nurse": "licensed practical nurse",
                "certified nursing assistant": "certified nursing assistant",
                "nurse practitioner": "nurse practitioner",
                "physician": "physician",
                "pharmacist": "pharmacist",
                
                # Facilities and Settings
                "long-term care": "long-term care facility",
                "nursing home": "nursing home",
                "assisted living": "assisted living facility",
                "memory care": "memory care unit",
                "skilled nursing": "skilled nursing facility",
                "rehabilitation center": "rehabilitation center"
            },
            
            "de": {
                # Conditions and Diseases
                "alzheimer": "Alzheimer-Krankheit",
                "dementia": "Demenz",
                "parkinson": "Parkinson-Krankheit",
                "diabetes": "Diabetes",
                "hypertension": "Bluthochdruck",
                "copd": "chronisch obstruktive Lungenerkrankung",
                "pneumonia": "Lungenentzündung",
                "uti": "Harnwegsinfektion",
                "dvt": "tiefe Venenthrombose",
                "stroke": "Schlaganfall",
                
                # Care Types
                "palliative": "Palliativpflege",
                "hospice": "Hospizpflege",
                "rehabilitation": "Rehabilitation",
                "physical therapy": "Physiotherapie",
                "occupational therapy": "Ergotherapie",
                "speech therapy": "Sprachtherapie",
                
                # Medical Procedures
                "wound care": "Wundversorgung",
                "medication administration": "Medikamentengabe",
                "vital signs": "Vitalzeichenüberwachung",
                "blood pressure": "Blutdruck",
                "glucose monitoring": "Blutzuckerüberwachung",
                "catheter care": "Katheterpflege",
                
                # Healthcare Roles
                "registered nurse": "examinierte Krankenschwester",
                "licensed practical nurse": "Krankenpflegehelferin",
                "certified nursing assistant": "Pflegeassistentin",
                "nurse practitioner": "Pflegefachkraft",
                "physician": "Arzt",
                "pharmacist": "Apotheker",
                
                # Facilities and Settings
                "long-term care": "Langzeitpflegeeinrichtung",
                "nursing home": "Pflegeheim",
                "assisted living": "betreutes Wohnen",
                "memory care": "Demenzstation",
                "skilled nursing": "Fachpflegeeinrichtung",
                "rehabilitation center": "Rehabilitationszentrum"
            },
            
            "fr": {
                # Conditions and Diseases
                "alzheimer": "maladie d'Alzheimer",
                "dementia": "démence",
                "parkinson": "maladie de Parkinson",
                "diabetes": "diabète",
                "hypertension": "hypertension",
                "copd": "bronchopneumopathie chronique obstructive",
                "pneumonia": "pneumonie",
                "uti": "infection urinaire",
                "dvt": "thrombose veineuse profonde",
                "stroke": "accident vasculaire cérébral",
                
                # Care Types
                "palliative": "soins palliatifs",
                "hospice": "soins de fin de vie",
                "rehabilitation": "réhabilitation",
                "physical therapy": "kinésithérapie",
                "occupational therapy": "ergothérapie",
                "speech therapy": "orthophonie",
                
                # Medical Procedures
                "wound care": "soins des plaies",
                "medication administration": "administration de médicaments",
                "vital signs": "surveillance des signes vitaux",
                "blood pressure": "tension artérielle",
                "glucose monitoring": "surveillance glycémique",
                "catheter care": "soins de cathéter",
                
                # Healthcare Roles
                "registered nurse": "infirmière diplômée",
                "licensed practical nurse": "infirmière auxiliaire",
                "certified nursing assistant": "aide-soignante",
                "nurse practitioner": "infirmière praticienne",
                "physician": "médecin",
                "pharmacist": "pharmacien",
                
                # Facilities and Settings
                "long-term care": "soins de longue durée",
                "nursing home": "maison de retraite",
                "assisted living": "résidence assistée",
                "memory care": "unité de soins mémoire",
                "skilled nursing": "soins infirmiers spécialisés",
                "rehabilitation center": "centre de réhabilitation"
            },
            
            "it": {
                # Conditions and Diseases
                "alzheimer": "malattia di Alzheimer",
                "dementia": "demenza",
                "parkinson": "malattia di Parkinson",
                "diabetes": "diabete",
                "hypertension": "ipertensione",
                "copd": "broncopneumopatia cronica ostruttiva",
                "pneumonia": "polmonite",
                "uti": "infezione del tratto urinario",
                "dvt": "trombosi venosa profonda",
                "stroke": "ictus",
                
                # Care Types
                "palliative": "cure palliative",
                "hospice": "cure hospice",
                "rehabilitation": "riabilitazione",
                "physical therapy": "fisioterapia",
                "occupational therapy": "terapia occupazionale",
                "speech therapy": "logopedia",
                
                # Medical Procedures
                "wound care": "cura delle ferite",
                "medication administration": "somministrazione farmaci",
                "vital signs": "monitoraggio segni vitali",
                "blood pressure": "pressione sanguigna",
                "glucose monitoring": "monitoraggio glicemia",
                "catheter care": "cura del catetere",
                
                # Healthcare Roles
                "registered nurse": "infermiere professionale",
                "licensed practical nurse": "infermiere generico",
                "certified nursing assistant": "operatore socio-sanitario",
                "nurse practitioner": "infermiere specialista",
                "physician": "medico",
                "pharmacist": "farmacista",
                
                # Facilities and Settings
                "long-term care": "assistenza a lungo termine",
                "nursing home": "casa di riposo",
                "assisted living": "residenza assistita",
                "memory care": "unità di cura della memoria",
                "skilled nursing": "assistenza infermieristica specializzata",
                "rehabilitation center": "centro di riabilitazione"
            }
        }
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Load common medical abbreviations and their expansions"""
        return {
            "BP": "blood pressure",
            "HR": "heart rate",
            "RR": "respiratory rate",
            "O2": "oxygen",
            "BG": "blood glucose",
            "UTI": "urinary tract infection",
            "COPD": "chronic obstructive pulmonary disease",
            "DVT": "deep vein thrombosis",
            "CVA": "cerebrovascular accident",
            "MI": "myocardial infarction",
            "CHF": "congestive heart failure",
            "DM": "diabetes mellitus",
            "HTN": "hypertension",
            "GERD": "gastroesophageal reflux disease",
            "ROM": "range of motion",
            "ADL": "activities of daily living",
            "PRN": "as needed",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "NPO": "nothing by mouth",
            "I&O": "intake and output"
        }
    
    def _load_drug_names(self) -> Set[str]:
        """Load common drug names that should be preserved during translation"""
        return {
            "acetaminophen", "ibuprofen", "aspirin", "warfarin", "heparin",
            "insulin", "metformin", "lisinopril", "amlodipine", "atorvastatin",
            "omeprazole", "furosemide", "metoprolol", "levothyroxine", "prednisone",
            "gabapentin", "tramadol", "morphine", "oxycodone", "lorazepam",
            "sertraline", "citalopram", "donepezil", "memantine", "risperidone"
        }
    
    def preprocess_healthcare_text(self, text: str, source_lang: str) -> str:
        """Preprocess healthcare text before translation"""
        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def postprocess_healthcare_translation(self, translated_text: str, target_lang: str) -> str:
        """Apply healthcare-specific post-processing to translations"""
        if target_lang not in self.terminology_db:
            return translated_text
        
        terms = self.terminology_db[target_lang]
        
        # Apply terminology corrections
        for en_term, target_term in terms.items():
            # Case-insensitive replacement with proper case preservation
            pattern = r'\b' + re.escape(en_term) + r'\b'
            
            def replace_func(match):
                original = match.group()
                if original.isupper():
                    return target_term.upper()
                elif original.istitle():
                    return target_term.title()
                else:
                    return target_term.lower()
            
            translated_text = re.sub(pattern, replace_func, translated_text, flags=re.IGNORECASE)
        
        return translated_text
    
    def extract_medical_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract medical entities from text (placeholder for NER integration)"""
        entities = []
        
        # Simple keyword-based extraction (in production, use spaCy NER or BioBERT)
        for lang_terms in self.terminology_db.values():
            for term in lang_terms.keys():
                if term.lower() in text.lower():
                    entities.append({
                        "text": term,
                        "label": "MEDICAL_TERM",
                        "start": text.lower().find(term.lower()),
                        "end": text.lower().find(term.lower()) + len(term)
                    })
        
        return entities
    
    def validate_medical_translation(self, original: str, translated: str, 
                                   source_lang: str, target_lang: str) -> Dict[str, float]:
        """Validate medical translation quality"""
        scores = {
            "terminology_accuracy": 0.0,
            "medical_entity_preservation": 0.0,
            "overall_medical_quality": 0.0
        }
        
        # Extract medical entities from both texts
        original_entities = self.extract_medical_entities(original)
        translated_entities = self.extract_medical_entities(translated)
        
        # Calculate terminology accuracy
        if original_entities:
            preserved_count = 0
            for entity in original_entities:
                # Check if equivalent term exists in translation
                if target_lang in self.terminology_db:
                    target_term = self.terminology_db[target_lang].get(entity["text"].lower())
                    if target_term and target_term.lower() in translated.lower():
                        preserved_count += 1
            
            scores["terminology_accuracy"] = preserved_count / len(original_entities)
        else:
            scores["terminology_accuracy"] = 1.0
        
        # Calculate entity preservation score
        if original_entities:
            scores["medical_entity_preservation"] = len(translated_entities) / len(original_entities)
        else:
            scores["medical_entity_preservation"] = 1.0
        
        # Overall medical quality score
        scores["overall_medical_quality"] = (
            scores["terminology_accuracy"] * 0.7 + 
            scores["medical_entity_preservation"] * 0.3
        )
        
        return scores

# Global instance
healthcare_processor = HealthcareTerminologyProcessor()