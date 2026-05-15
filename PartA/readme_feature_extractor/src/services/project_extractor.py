import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
import instructor

# =========================
# 1. Schema
# =========================
class ProjectFeatures(BaseModel):
    project_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tech_stack: List[str] = Field(default_factory=list)
    installation_commands: List[str] = Field(default_factory=list)
    has_tests: bool = Field(default=False)
    license_type: Optional[str] = Field(default=None)


# =========================
# 2. Regex Helpers (FAST ⚡)
# =========================
def extract_tech_stack_fast(text: str) -> List[str]:
    # قائمة بأشهر التقنيات عشان نصطادها بسرعة من غير ما نستنى الموديل
    known = [
        "python", "django", "flask", "fastapi",
        "react", "vue", "angular", "next.js", "node.js",
        "tensorflow", "pytorch", "scikit-learn",
        "docker", "kubernetes", "sql", "postgresql", "mongodb"
    ]
    return [k for k in known if k in text.lower()]


def detect_tests_fast(text: str) -> bool:
    keywords = ["pytest", "unittest", "tests/", "coverage", "tox", "npm test", "jest"]
    return any(k in text.lower() for k in keywords)


def detect_license_fast(text: str) -> Optional[str]:
    licenses = ["MIT", "Apache", "GPL", "BSD", "Mozilla", "Creative Commons"]
    for lic in licenses:
        if lic.lower() in text.lower():
            return lic
    return None


# =========================
# 3. Smart Section Builder
# =========================
def build_important_text(cleaned_readme: dict) -> str:
    # 💡 التعديل المهم: سحب الكلام والأكواد من القاموس اللي الـ Cleaner بيبعته
    prose = cleaned_readme.get("prose", "")
    code_blocks = "\n".join(cleaned_readme.get("code_blocks", []))
    
    return prose + "\n\n" + code_blocks


# =========================
# 4. Confidence Scoring
# =========================
def confidence_score(value, text):
    if not value:
        return 0.3
    if isinstance(value, list):
        return min(0.95, 0.5 + 0.1 * len(value))
    if str(value).lower() in text.lower():
        return 0.9
    return 0.6


# =========================
# 5. Main Extraction (Hybrid)
# =========================
def extract_features(cleaned_readme: dict, requirements_text: str, model) -> str:
    
    # 🔹 Step 1: Preprocess (FAST)
    important_text = build_important_text(cleaned_readme)
    full_text = important_text + "\n" + requirements_text

    # 🔹 Step 2: Fast extraction (No LLM) - بيخلص في أجزاء من الثانية
    tech_fast = extract_tech_stack_fast(full_text)
    has_tests_fast = detect_tests_fast(full_text)
    license_fast = detect_license_fast(full_text)

    # 🔹 Step 3: LLM Extraction (ONLY for complex fields)
    patched_create = instructor.patch(
        create=model.create_chat_completion_openai_v1,
        mode=instructor.Mode.MD_JSON,
    )

    prompt = f"""
You are a strict software analyzer.

Rules:
- Do NOT hallucinate
- If missing → return null or empty
- Be concise

Extract ONLY:
- project_name
- description (max 2 sentences)
- installation_commands (ONLY terminal commands)

TEXT:
{full_text}
"""

    try:
        print("DEBUG - Asking LLM to extract complex features...")
        extracted: ProjectFeatures = patched_create(
            response_model=ProjectFeatures,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2, # قللناه عشان يبقى مركز ومايألفش
            max_tokens=800,  # مش محتاجين كتير لأنه بيجيب حاجات صغيرة
            max_retries=2,
        )
        print(f"DEBUG - LLM Output successfully generated!")
    except Exception as e:
        print(f"DEBUG - Extraction Error: {str(e)}")
        # لو الموديل ضرب لأي سبب، هنرجع اللي جبناه بالـ Regex عشان ميبقاش فاضي خالص
        extracted = ProjectFeatures(
            tech_stack=tech_fast,
            has_tests=has_tests_fast,
            license_type=license_fast
        )

    # 🔹 Step 4: Merge (Hybrid 💡) - دمج ذكاء الموديل مع سرعة الـ Regex
    extracted.tech_stack = list(set(extracted.tech_stack + tech_fast))
    extracted.has_tests = has_tests_fast or extracted.has_tests
    extracted.license_type = extracted.license_type or license_fast

    # 🔹 Step 5: Validation
    if extracted.description and len(extracted.description) > 300:
        extracted.description = extracted.description[:300]

    # 🔹 Step 6: Confidence
    confidence = {
        "project_name": confidence_score(extracted.project_name, full_text),
        "description": confidence_score(extracted.description, full_text),
        "tech_stack": confidence_score(extracted.tech_stack, full_text),
        "installation_commands": confidence_score(extracted.installation_commands, full_text),
        "has_tests": 0.9 if extracted.has_tests else 0.6,
        "license_type": confidence_score(extracted.license_type, full_text),
    }

    low_conf = [k for k, v in confidence.items() if v < 0.5]

    # 🔹 Step 7: Final Output
    output = {
        "features": extracted.model_dump(),
        "confidence": confidence,
        "low_confidence_fields": low_conf
    }

    return json.dumps(output, indent=2, ensure_ascii=False)