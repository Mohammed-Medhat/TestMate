# TestMate: AI-Powered Automated Feature Extraction & Test Generation

## 🏗️ System Architecture
The system follows a Clean Architecture design to ensure scalability, modularity, and maintainability.

### 🔹 Backend (FastAPI)
Handles AI inference and feature extraction.
* **Powered by:** Qwen 2.5 Coder (7B) for deep technical understanding.
* **Performs:**
  * README parsing
  * Entity extraction
  * Test scenario generation
* **Storage:**
  * **SQLite:** Logs and system operations.
  * **MongoDB:** Structured feedback and refinement loop.

### 🔹 Frontend (Next.js)
Modern Cyberpunk-style dashboard using Tailwind CSS.
* **Provides:**
  * 🔄 **Real-time pipeline visualization:** (Fetching → Cleaning → AI Processing)
  * 📊 **Interactive Results Dashboard**
  * ✏️ **Manual editing and refinement interface**

---

## ⚙️ Prerequisites
Before running the project, ensure you have the following installed:
* **Python** 3.10+
* **Node.js** 18+ (LTS recommended)
* **MongoDB** (Local or Cloud)

**Hardware Requirements:**
* **Minimum:** 8GB RAM
* **Recommended:** 16GB RAM (for smooth LLM inference on CPU)

---

## 📥 Installation & Setup

### 1️⃣ Backend Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
⚠️ AI Model Setup (Required)
The model file is NOT included in the repository due to its large size (~4.7GB).

Option 1: Automated Download

Bash
python download_model.py
Option 2: Manual Download

Download the model: qwen2.5-coder-7b-instruct-q4_k_m.gguf

Place it inside the models/ directory in the root folder.

###2️⃣ Frontend Setup
Bash
# Navigate to the frontend directory
cd readme_extractor

# Install Node modules
npm install

# Start the development server
npm run dev
🧠 Key Features
🔍 Automated Feature Extraction: Converts raw README files into structured JSON formats, intelligently extracting the Project description, Tech stack, and Dependencies.

🧪 Test Scenario Generation: Automatically generates comprehensive test cases:

✅ Positive test cases

❌ Negative test cases

⚠️ Edge cases

📊 Confidence Scoring: Assigns a reliability score for each extracted field, helping developers assess the AI output quality at a glance.

🔁 Refinement Loop: Users can seamlessly edit extracted data via the UI. These changes are saved in MongoDB to establish a feedback loop that improves future AI performance.