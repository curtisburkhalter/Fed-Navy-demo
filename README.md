# Maritime Surveillance Intelligence Generator

**Powered by HP ZGX Nano AI Station**

A Vision Language Model (VLM) demonstration for US Navy reconnaissance imagery analysis with synthetic geolocation and structured intelligence report generation.

---

## Overview

This demo showcases on-premises AI capability for maritime surveillance imagery analysis. The system:

- Analyzes reconnaissance imagery using **BLIP-2 FLAN-T5-XL** vision-language model
- Generates synthetic geolocation data within configurable operating regions
- Produces structured intelligence reports with threat assessments
- Runs entirely on-premises with no cloud dependency

### Key Features

| Feature | Description |
|---------|-------------|
| **Visual Question Answering** | BLIP-2 answers specific questions about vessel type, cargo, activity, size, and heading |
| **Threat Assessment** | Automatic threat level classification (NONE/LOW/MEDIUM/HIGH/CRITICAL) based on vessel and cargo analysis |
| **Synthetic Geolocation** | Realistic coordinates, MGRS grid references, and relative positions for 8 operating regions |
| **Contextual Recommendations** | Dynamic recommendations based on threat level, vessel category, and detected cargo |
| **Hybrid Template + LLM** | Structured report format with LLM-generated natural language content |

### Value Propositions

| Benefit | Description |
|---------|-------------|
| **Data Security** | Classified imagery never leaves the secure environment |
| **Zero Latency** | Real-time analysis without network round-trips |
| **Operational Independence** | Functions without satellite connectivity (ships at sea) |
| **Cost Efficiency** | No per-query API costs or cloud subscriptions |

---

## Architecture

### Models Used

| Model | Purpose | Size |
|-------|---------|------|
| **Salesforce/blip2-flan-t5-xl** | Visual Question Answering - extracts vessel type, cargo, activity, size, heading | ~15GB |
| **TinyLlama/TinyLlama-1.1B-Chat-v1.0** | Text generation for report enhancement | ~2GB |

### Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript with military-themed UI
- **ML Framework**: Hugging Face Transformers
- **Inference**: PyTorch with CUDA support

---

## Quick Start

### Prerequisites

- HP ZGX Nano AI Station (or NVIDIA GPU with 24GB+ VRAM)
- Python 3.10+
- CUDA 12.0+

### Installation

```bash
# Clone or copy the demo files to your ZGX
cd /home/your_username/Desktop
mkdir Fed-Navy-demo
cd Fed-Navy-demo

# Create virtual environment
python3 -m venv navy-env
source navy-env/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate fastapi uvicorn python-multipart Pillow

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Directory Structure

```
Fed-Navy-demo/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Web interface
├── models/                   # (Auto-downloaded on first run)
├── navy-env/                 # Python virtual environment
├── start_demo_remote.sh     # Remote access startup script
├── install.sh               # Installation script
└── README.md                # This file
```

### Starting the Demo

**Local access:**
```bash
cd /home/your_username/Desktop/Fed-Navy-demo
source navy-env/bin/activate
python3 backend/main.py
# Open http://localhost:8000
```

**Remote access (from Windows laptop):**
```bash
./start_demo_remote.sh
```

The script will display the server IP and access URL:
```
======================================
✅ Demo is running!
======================================

Access the demo from your Windows laptop:
👉 http://192.168.x.x:8000

Backend API endpoints:
  - Status: http://192.168.x.x:8000/
  - Health: http://192.168.x.x:8000/api/health
  - Regions: http://192.168.x.x:8000/api/regions

Instructions:
1. Open the web interface in your browser
2. Select image to analyze
3. Click Analyze Image

⚠️  Note: Models use ~17GB total. Ensure sufficient GPU memory!

Press Ctrl+C to stop the demo
======================================
```

Simply click the URL shown or copy it to your Windows browser.

### First Run

On first startup, the models will be downloaded from Hugging Face (~17GB total). This may take 10-15 minutes depending on network speed. Subsequent starts will be much faster as models are cached.

---

## Demo Features

### Operating Regions

The system supports 8 Areas of Responsibility (AORs) with realistic coordinate generation:

| Region | Coverage | Key Landmarks |
|--------|----------|---------------|
| **Western Pacific** | Philippine Sea, Taiwan Strait | East China Sea, Taiwan Strait |
| **South China Sea** | Spratly/Paracel Islands | Gulf of Tonkin |
| **Arabian Gulf** | Persian Gulf region | Strait of Hormuz, Gulf of Oman |
| **Gulf of Aden** | Horn of Africa | Bab el-Mandeb Strait, Djibouti |
| **Eastern Mediterranean** | Cyprus, Crete | Suez Canal approaches |
| **North Atlantic** | Trans-Atlantic routes | GIUK Gap, Azores |
| **Indo-Pacific** | Southeast Asia straits | Malacca Strait, Singapore Strait |
| **California Coast** | Training region | San Diego, Monterey Bay |

### Intelligence Report Format

Each analysis generates a structured report with 6 sections:

```
1. VESSEL CLASSIFICATION
   Type: [Vessel type from BLIP-2]
   Category: [COMMERCIAL/MILITARY/GOVERNMENT classification]

2. PHYSICAL CHARACTERISTICS
   Estimated Size: [Small/Medium/Large]
   Cargo/Payload: [Detected cargo or equipment]
   Visual Description: [BLIP-2 image description]

3. ACTIVITY ASSESSMENT
   Current Status: [Moving/Anchored/Docked]
   Heading: [Direction of travel]
   Observation: [LLM-generated activity description]

4. THREAT ASSESSMENT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   THREAT LEVEL: [NONE/LOW/MEDIUM/HIGH/CRITICAL]
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Justification: [Explanation for threat level]

5. CONFIDENCE LEVEL: HIGH
   Assessment based on BLIP-2 visual question answering

6. RECOMMENDATIONS
   - [Contextual recommendations based on threat/cargo]
   - Cross-reference with AIS/maritime tracking data
   - Verify vessel registry and flag state
```

### Threat Level Classification

| Level | Triggers | Example Vessels |
|-------|----------|-----------------|
| **NONE** | Commercial cargo, containers, merchant vessels | Container ships, bulk carriers |
| **LOW** | Tankers, fishing vessels, unidentified | Oil tankers, trawlers |
| **MEDIUM** | Government vessels, foreign national flags | Coast guard, patrol boats |
| **HIGH** | Military surface combatants | Destroyers, frigates, cruisers |
| **CRITICAL** | Submarines | Any subsurface contact |

### Contextual Recommendations

Recommendations are dynamically generated based on:

**By Threat Level:**
- **CRITICAL**: Alert fleet command, continuous tracking, prepare ASW assets
- **HIGH**: Alert regional command, visual ID of hull number, monitor for weapons activation
- **MEDIUM**: Increase monitoring frequency, coordinate with allied assets
- **NONE/LOW**: Continue routine monitoring

**By Cargo/Payload Detection:**
- **Military/Warship**: Assess military cargo and potential threat capability
- **Foreign National**: Flag for foreign national vessel monitoring program
- **Containers**: Verify cargo manifest against sanctions lists
- **Oil/Tanker**: Monitor for sanctions evasion or illegal transfer
- **Fishing**: Check for illegal fishing or maritime militia indicators

---

## Stakeholder Presentation Guide

### S.T.A.R. Framework

**Situation:**
Naval reconnaissance generates massive volumes of imagery from UAVs, satellites, and manned aircraft. Analysts are overwhelmed with imagery backlogs, and classified data cannot be processed via cloud services.

**Task:**
Enable rapid, secure imagery analysis that:
- Processes images in seconds, not hours
- Maintains SCIF-level data isolation
- Functions in disconnected environments (ships at sea)

**Action:**
Deploy HP ZGX Nano AI Station with on-premises VLM capability:
- BLIP-2 FLAN-T5-XL for visual question answering
- TinyLlama for natural language report generation
- GPU-accelerated processing on NVIDIA Grace Blackwell

**Result:**
- Image analysis in 5-10 seconds vs. manual review (10-30 minutes)
- 100% data security - imagery never leaves the platform
- Operational in degraded/disconnected network conditions
- Automatic threat classification and contextual recommendations

### Demo Script

1. **Introduction** (1 min)
   - Show the clean, military-themed interface
   - Highlight "UNCLASSIFIED // FOR DEMO" classification banner
   - Point out HP ZGX Nano branding

2. **Upload Test Image** (1 min)
   - Use a maritime/vessel image (cargo ship, naval vessel, etc.)
   - Select an operationally relevant AOR (e.g., South China Sea, Western Pacific)
   - Click "Generate Intelligence Report"

3. **Review Output** (2-3 min)
   - Walk through the 6-section report format
   - Highlight the threat level and justification
   - Show how recommendations change based on detected threats
   - Note the synthetic coordinates and grid references

4. **Value Discussion** (2 min)
   - Emphasize processing speed (seconds vs. hours)
   - Discuss data security implications
   - Note operational independence (no cloud required)
   - Address scalability and deployment options

### Suggested Test Images

For compelling demonstrations, use images containing:
- **Commercial vessels**: Container ships, tankers, bulk carriers
- **Naval vessels**: Destroyers, frigates (use unclassified reference photos)
- **Fishing fleets**: Multiple small vessels in formation
- **Port facilities**: Ships at dock with visible cargo operations
- **Aerial perspectives**: Overhead/satellite-style imagery

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/api/health` | GET | Health check and model status |
| `/api/regions` | GET | List available operating regions |
| `/api/analyze` | POST | Analyze image and generate report |

### POST /api/analyze

**Request:**
- `image`: Image file (multipart/form-data)
- `region`: Operating region ID (default: "western_pacific")
- `custom_instructions`: Additional analysis guidance (optional)

**Response:**
```json
{
  "report_id": "SURV-20260112-1234",
  "classification": "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY",
  "capture_time": "12 JAN 2026 2345Z",
  "location": {
    "coordinates_dms": "21°49'59\"N, 142°43'29\"E",
    "coordinates_decimal": {"lat": 21.8332, "lon": 142.724997},
    "grid_reference": "54Q DD 8885 7602",
    "relative_position": "135nm 43° from Taiwan Strait",
    "operating_region": "Western Pacific"
  },
  "analysis": "...[structured report text]...",
  "raw_analysis": {
    "vessel_type": "a cargo ship",
    "description": "a ship in the water",
    "cargo": "containers",
    "activity": "moving through the water",
    "size": "large ship",
    "heading": "moving in a straight line"
  },
  "generated_at": "2026-01-12T23:45:00Z"
}
```

---

## Troubleshooting

### Model Loading Errors

```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Clear Hugging Face cache and re-download
rm -rf ~/.cache/huggingface/
python3 backend/main.py
```

### Out of Memory Errors

BLIP-2 FLAN-T5-XL requires ~15GB VRAM. If you encounter OOM errors:

```python
# In main.py, try loading with 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    quantization_config=quantization_config
)
```

### Port Already in Use

```bash
# Find and kill existing process
lsof -i :8000
kill -9 <PID>
```

### Remote Access Issues

```bash
# Check server IP
hostname -I

# Verify server is listening on all interfaces
netstat -tlnp | grep 8000

# Test local connection first
curl http://localhost:8000/api/health
```

---

## Technical Specifications

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| System RAM | 32GB | 64GB |
| Storage | 50GB | 100GB SSD |
| GPU | NVIDIA Ampere+ | NVIDIA Grace Blackwell (GB10) |

### Model Performance

| Metric | Value |
|--------|-------|
| Image Analysis Time | 5-10 seconds |
| Report Generation | 2-3 seconds |
| Total Processing | ~10-15 seconds |
| Throughput | ~4-6 images/minute |

---

## Security Notice

This demonstration generates **synthetic geolocation data** for illustration purposes. No actual operational coordinates are used or inferred from imagery.

The classification banner displays `UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY` to clearly indicate the demo nature of the application.

**For operational deployment:**
- Implement proper access controls and authentication
- Enable audit logging for all analysis requests
- Integrate with existing classification marking systems
- Review and approve all model outputs before operational use

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release with LLaVA model |
| 1.1 | Jan 2026 | Switched to Hugging Face Transformers pipeline |
| 1.2 | Jan 2026 | Upgraded to BLIP for better image captioning |
| 2.0 | Jan 2026 | Major upgrade to BLIP-2 FLAN-T5-XL with VQA |
| 2.1 | Jan 2026 | Added hybrid template + LLM report generation |
| 2.2 | Jan 2026 | Contextual threat assessment and recommendations |

---

## Support

For technical assistance or demo customization, contact the HP ZGX Product Team.

**Powered by HP ZGX Nano AI Station**
