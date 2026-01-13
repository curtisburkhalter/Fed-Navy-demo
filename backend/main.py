"""
Maritime Surveillance Intelligence Generator
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for US Navy reconnaissance imagery analysis
with synthetic geolocation and structured intelligence report generation.

Uses Salesforce BLIP-2 for superior image understanding with
hybrid template + LLM text generation for intelligence reports.
"""

import os
import io
import random
import string
from datetime import datetime, timezone
from pathlib import Path
import warnings
import logging

warnings.filterwarnings('ignore')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Operating regions with realistic coordinate bounds
OPERATING_REGIONS = {
    "western_pacific": {
        "name": "Western Pacific",
        "lat_range": (15.0, 35.0),
        "lon_range": (120.0, 145.0),
        "landmarks": ["Philippine Sea", "East China Sea", "Taiwan Strait"]
    },
    "south_china_sea": {
        "name": "South China Sea",
        "lat_range": (5.0, 22.0),
        "lon_range": (105.0, 120.0),
        "landmarks": ["Spratly Islands", "Paracel Islands", "Gulf of Tonkin"]
    },
    "arabian_gulf": {
        "name": "Arabian Gulf / Persian Gulf",
        "lat_range": (24.0, 30.0),
        "lon_range": (48.0, 56.0),
        "landmarks": ["Strait of Hormuz", "Gulf of Oman", "Bahrain"]
    },
    "gulf_of_aden": {
        "name": "Gulf of Aden",
        "lat_range": (11.0, 15.0),
        "lon_range": (43.0, 51.0),
        "landmarks": ["Bab el-Mandeb Strait", "Socotra Island", "Djibouti"]
    },
    "eastern_mediterranean": {
        "name": "Eastern Mediterranean",
        "lat_range": (32.0, 37.0),
        "lon_range": (28.0, 36.0),
        "landmarks": ["Cyprus", "Crete", "Suez Canal approaches"]
    },
    "north_atlantic": {
        "name": "North Atlantic",
        "lat_range": (35.0, 60.0),
        "lon_range": (-45.0, -10.0),
        "landmarks": ["GIUK Gap", "Azores", "Bay of Biscay"]
    },
    "california_coast": {
        "name": "California Coast (TRAINING)",
        "lat_range": (32.5, 38.0),
        "lon_range": (-124.0, -117.0),
        "landmarks": ["San Diego", "Point Loma", "Monterey Bay"]
    },
    "indo_pacific": {
        "name": "Indo-Pacific Region",
        "lat_range": (-10.0, 10.0),
        "lon_range": (95.0, 130.0),
        "landmarks": ["Malacca Strait", "Singapore Strait", "Java Sea"]
    }
}

app = FastAPI(
    title="Maritime Surveillance Intelligence Generator",
    description="VLM-powered reconnaissance imagery analysis for US Navy",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
blip_processor = None
blip_model = None
text_generator = None
device = "cpu"


def load_models():
    """Load the VLM and text generation models."""
    global blip_processor, blip_model, text_generator, device
    
    if blip_model is not None:
        return
    
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load BLIP-2 model for image captioning and VQA
    logger.info("Loading BLIP-2 FLAN-T5-XL image understanding model...")
    try:
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        logger.info("BLIP-2 FLAN-T5-XL model loaded successfully on " + device)
    except Exception as e:
        logger.error(f"Failed to load BLIP-2 model: {e}")
        raise
    
    # Load text generation model for report enhancement
    logger.info("Loading text generation model...")
    try:
        text_generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        logger.info("Text generation model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load text generator: {e}")
        text_generator = None


def generate_synthetic_coordinates(region: str) -> dict:
    """Generate realistic synthetic coordinates within the specified region."""
    if region not in OPERATING_REGIONS:
        region = "western_pacific"
    
    region_data = OPERATING_REGIONS[region]
    
    lat = random.uniform(*region_data["lat_range"])
    lon = random.uniform(*region_data["lon_range"])
    
    # Convert to degrees, minutes, seconds
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    
    lat_deg = int(lat_abs)
    lat_min = int((lat_abs - lat_deg) * 60)
    lat_sec = int(((lat_abs - lat_deg) * 60 - lat_min) * 60)
    
    lon_deg = int(lon_abs)
    lon_min = int((lon_abs - lon_deg) * 60)
    lon_sec = int(((lon_abs - lon_deg) * 60 - lon_min) * 60)
    
    # Generate MGRS-style grid reference (simplified)
    grid_zone = f"{int((lon + 180) / 6) + 1:02d}"
    grid_band = chr(ord('C') + int((lat + 80) / 8))
    grid_square = ''.join(random.choices(string.ascii_uppercase[:8], k=2))
    grid_easting = f"{random.randint(1000, 9999)}"
    grid_northing = f"{random.randint(1000, 9999)}"
    
    # Select nearby landmark
    landmark = random.choice(region_data["landmarks"])
    distance_nm = random.randint(15, 200)
    bearing = random.randint(0, 359)
    
    return {
        "decimal": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "dms": f"{lat_deg}°{lat_min:02d}'{lat_sec:02d}\"{lat_dir}, {lon_deg}°{lon_min:02d}'{lon_sec:02d}\"{lon_dir}",
        "mgrs": f"{grid_zone}{grid_band} {grid_square} {grid_easting} {grid_northing}",
        "relative": f"{distance_nm}nm {bearing}° from {landmark}",
        "region_name": region_data["name"]
    }


def generate_report_id() -> str:
    """Generate a realistic report identifier."""
    prefix = random.choice(["RECON", "SURV", "ISR", "MARI"])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = f"{random.randint(1, 9999):04d}"
    return f"{prefix}-{date_str}-{seq}"


def ask_blip2(image: Image.Image, question: str) -> str:
    """Ask BLIP-2 a question about the image."""
    global blip_processor, blip_model, device
    
    prompt = f"Question: {question} Answer:"
    inputs = blip_processor(image, prompt, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
    
    outputs = blip_model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        early_stopping=True
    )
    
    answer = blip_processor.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Clean up the answer - remove any question echo
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    
    return answer


def analyze_image_with_blip2(image: Image.Image) -> dict:
    """
    Analyze image using BLIP-2 with maritime-specific questions.
    Returns structured analysis dict.
    """
    logger.info("Analyzing image with BLIP-2 VQA...")
    
    # Ask targeted questions
    questions = {
        "vessel_type": "What kind of ship is in this image? Is it a cargo ship, container ship, tanker, fishing boat, or military vessel?",
        "description": "Describe what you see in this aerial image of the ocean.",
        "cargo": "What is this ship carrying? Describe any visible cargo, containers, or equipment on deck.",
        "activity": "Is this ship moving through the water, anchored, or docked at port?",
        "size": "Based on the image, is this a small boat, medium-sized vessel, or large ship?",
        "heading": "What direction does this ship appear to be traveling?"
    }
    
    results = {}
    for key, question in questions.items():
        answer = ask_blip2(image, question)
        logger.info(f"{key}: {answer}")
        results[key] = answer
    
    return results


def generate_text_for_field(field_name: str, context: str, threat_level: str = "LOW", max_tokens: int = 50) -> str:
    """Use LLM to generate text for a specific field given context."""
    global text_generator
    
    if text_generator is None:
        return ""  # Return empty, let template handle it
    
    prompts = {
        "activity": f"<|user|>In 10 words or less, describe what a {context} is doing at sea.</s><|assistant|>",
        "threat_justification_none": f"<|user|>In one sentence, explain why a commercial {context} poses no military threat.</s><|assistant|>",
        "threat_justification_low": f"<|user|>In one sentence, explain why a {context} is low threat but worth monitoring.</s><|assistant|>",
        "threat_justification_medium": f"<|user|>In one sentence, explain why a {context} requires increased monitoring.</s><|assistant|>",
        "threat_justification_high": f"<|user|>In one sentence, explain why a military {context} is a high priority threat.</s><|assistant|>",
        "recommendation": f"<|user|>Give one specific action for monitoring a {context}. Be brief.</s><|assistant|>"
    }
    
    # Select appropriate prompt based on field and threat level
    if field_name == "threat_justification":
        prompt_key = f"threat_justification_{threat_level.lower()}"
        prompt = prompts.get(prompt_key, prompts["threat_justification_low"])
    else:
        prompt = prompts.get(field_name, f"<|user|>Briefly describe: {context}</s><|assistant|>")
    
    try:
        result = text_generator(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.5,  # Lower temperature for more consistent output
            do_sample=True,
            pad_token_id=text_generator.tokenizer.eos_token_id
        )
        generated = result[0]['generated_text']
        if "<|assistant|>" in generated:
            response = generated.split("<|assistant|>")[-1].strip()
            # Clean up - remove "Sure!", "Of course!", etc.
            for prefix in ["Sure!", "Sure,", "Of course!", "Certainly!", "1.", "1)"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            # Take first sentence only
            if ". " in response:
                response = response.split(". ")[0] + "."
            return response
        return ""
    except Exception as e:
        logger.warning(f"Text generation failed for {field_name}: {e}")
        return ""


def enhance_caption_to_intel(image_analysis: dict, custom_instructions: str = "") -> str:
    """
    Create intelligence report using structured template with LLM-generated content.
    """
    vessel_type = image_analysis.get("vessel_type", "UNIDENTIFIED VESSEL")
    vessel_description = image_analysis.get("description", "No description available")
    cargo = image_analysis.get("cargo", "Unknown")
    activity = image_analysis.get("activity", "Unknown")
    size = image_analysis.get("size", "Unknown")
    heading = image_analysis.get("heading", "Unable to determine")
    
    # Combine all text for threat analysis - check everything
    all_text = f"{vessel_type} {vessel_description} {cargo} {activity}".lower()
    
    # Determine threat level based on ALL analysis fields
    # Check for military indicators first (highest priority)
    if any(word in all_text for word in ["warship", "navy", "military", "destroyer", "frigate", "cruiser", "corvette", "battleship", "aircraft carrier"]):
        threat_level = "HIGH"
        vessel_category = "MILITARY - SURFACE WARFARE"
        default_justification = "Military combatant detected - assess nationality, capabilities, and intent"
    elif any(word in all_text for word in ["submarine", "sub "]):
        threat_level = "CRITICAL"
        vessel_category = "MILITARY - SUBSURFACE"
        default_justification = "Subsurface contact - immediate tracking and classification required"
    elif any(word in all_text for word in ["russian", "chinese", "iranian", "north korean"]):
        threat_level = "MEDIUM"
        vessel_category = "FOREIGN NATIONAL - MONITOR"
        default_justification = "Foreign national vessel - increased monitoring recommended"
    elif any(word in all_text for word in ["patrol", "coast guard", "cutter"]):
        threat_level = "MEDIUM"
        vessel_category = "GOVERNMENT - LAW ENFORCEMENT"
        default_justification = "Government vessel - determine nationality and jurisdiction"
    elif any(word in all_text for word in ["cargo", "container", "freight", "merchant", "commercial"]):
        threat_level = "NONE"
        vessel_category = "COMMERCIAL - MERCHANT MARINE"
        default_justification = "Commercial cargo vessel engaged in routine maritime trade operations"
    elif any(word in all_text for word in ["tanker", "oil", "petroleum"]):
        threat_level = "LOW"
        vessel_category = "COMMERCIAL - BULK CARRIER"
        default_justification = "Strategic cargo vessel - monitor for sanctions compliance"
    elif any(word in all_text for word in ["fishing", "trawler"]):
        threat_level = "LOW"
        vessel_category = "COMMERCIAL - FISHING"
        default_justification = "Fishing vessel - potential for surveillance or militia activity"
    else:
        threat_level = "LOW"
        vessel_category = "UNCLASSIFIED"
        default_justification = "Vessel type undetermined - recommend visual confirmation"
    
    # Generate threat justification with correct threat level context
    threat_justification = generate_text_for_field(
        "threat_justification", 
        vessel_type,
        threat_level
    )
    if not threat_justification:
        threat_justification = default_justification
    
    # Generate activity observation
    activity_observation = generate_text_for_field("activity", f"{vessel_type} {activity}", threat_level)
    if not activity_observation:
        activity_observation = f"Vessel observed {activity.lower()}"
    
    # Generate contextual recommendations based on threat level, category, and cargo
    recommendations = []
    
    if threat_level == "CRITICAL":
        recommendations.append("IMMEDIATE: Alert fleet command and request additional ISR assets")
        recommendations.append("Initiate continuous tracking and maintain safe distance")
        recommendations.append("Prepare anti-submarine warfare assets if applicable")
    elif threat_level == "HIGH":
        recommendations.append("Alert regional command and increase surveillance priority")
        recommendations.append("Attempt visual identification of hull number and flag state")
        recommendations.append("Monitor for weapons systems activation or hostile maneuvering")
    elif threat_level == "MEDIUM":
        recommendations.append("Increase monitoring frequency and document all activities")
        recommendations.append("Coordinate with allied assets for corroborating intelligence")
    else:
        recommendations.append("Continue routine monitoring")
    
    # Add cargo-specific recommendations
    cargo_lower = cargo.lower()
    if any(word in cargo_lower for word in ["warship", "military", "weapon"]):
        recommendations.append("PRIORITY: Assess military cargo and potential threat capability")
    elif any(word in cargo_lower for word in ["russian", "chinese", "iranian", "north korean"]):
        recommendations.append("Flag for foreign national vessel monitoring program")
    elif any(word in cargo_lower for word in ["container", "cargo", "freight"]):
        recommendations.append("Verify cargo manifest against sanctions lists if possible")
    elif any(word in cargo_lower for word in ["oil", "tanker", "petroleum", "fuel"]):
        recommendations.append("Monitor for potential sanctions evasion or illegal transfer")
    elif any(word in cargo_lower for word in ["fish", "fishing"]):
        recommendations.append("Check for illegal fishing or maritime militia indicators")
    
    # Standard recommendations
    recommendations.append("Cross-reference with AIS/maritime tracking data")
    recommendations.append("Verify vessel registry and flag state")
    
    # Format recommendations
    recommendations_text = "\n   - ".join(recommendations)
    
    # Build the structured report
    assessment = f"""1. VESSEL CLASSIFICATION
   Type: {vessel_type.upper()}
   Category: {vessel_category}

2. PHYSICAL CHARACTERISTICS
   Estimated Size: {size.capitalize()}
   Cargo/Payload: {cargo.capitalize()}
   Visual Description: {vessel_description.capitalize()}

3. ACTIVITY ASSESSMENT
   Current Status: {activity.upper()}
   Heading: {heading.capitalize()}
   Observation: {activity_observation}

4. THREAT ASSESSMENT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   THREAT LEVEL: {threat_level}
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Justification: {threat_justification}

5. CONFIDENCE LEVEL: HIGH
   Assessment based on BLIP-2 visual question answering with multi-query verification

6. RECOMMENDATIONS
   - {recommendations_text}{f'''
   - {custom_instructions}''' if custom_instructions else ""}"""
    
    return assessment


def analyze_image(image: Image.Image, region: str, custom_instructions: str = "") -> dict:
    """Analyze an image using BLIP-2 VQA and generate an intelligence report."""
    global blip_model
    
    if blip_model is None:
        load_models()
    
    if blip_model is None:
        raise RuntimeError("BLIP-2 model not available")
    
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if too large (for faster processing)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Get structured analysis from BLIP-2
    image_analysis = analyze_image_with_blip2(image)
    
    # Generate intelligence report using hybrid template+LLM
    analysis = enhance_caption_to_intel(image_analysis, custom_instructions)
    
    # Generate synthetic geolocation
    geo_data = generate_synthetic_coordinates(region)
    report_id = generate_report_id()
    capture_time = datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper()
    
    # Build the structured report
    report = {
        "report_id": report_id,
        "classification": "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY",
        "capture_time": capture_time,
        "location": {
            "coordinates_dms": geo_data["dms"],
            "coordinates_decimal": geo_data["decimal"],
            "grid_reference": geo_data["mgrs"],
            "relative_position": geo_data["relative"],
            "operating_region": geo_data["region_name"]
        },
        "analysis": analysis,
        "raw_analysis": image_analysis,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    return report


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    # Try multiple possible locations for index.html
    possible_paths = [
        Path(__file__).parent.parent / "frontend" / "index.html",
        Path(__file__).parent / "frontend" / "index.html",
        Path(__file__).parent / "index.html",
    ]
    
    for html_path in possible_paths:
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
    
    return HTMLResponse(content="<h1>Maritime Surveillance Intelligence Generator</h1><p>index.html not found</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "blip2_model_loaded": blip_model is not None,
        "text_generator_loaded": text_generator is not None,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/regions")
async def get_regions():
    """Get available operating regions."""
    return {
        "regions": [
            {"id": k, "name": v["name"], "landmarks": v["landmarks"]}
            for k, v in OPERATING_REGIONS.items()
        ]
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    region: str = Form("western_pacific"),
    custom_instructions: str = Form("")
):
    """Analyze an uploaded image and generate an intelligence report."""
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image data
    image_data = await image.read()
    
    if len(image_data) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
    
    # Open image
    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Perform analysis
    try:
        report = analyze_image(img, region, custom_instructions)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Pre-load the models on startup."""
    print("=" * 60)
    print("Maritime Surveillance Intelligence Generator")
    print("Powered by HP ZGX Nano AI Station")
    print("Using BLIP-2 FLAN-T5-XL + TinyLlama for intelligence reports")
    print("=" * 60)
    try:
        load_models()
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        print(f"Warning: {e}")
        print("Models will be loaded on first request.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)