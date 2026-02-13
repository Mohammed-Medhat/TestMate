"""
model_runner_ALTERNATIVE.py
Alternative approach: Load adapter with explicit device handling
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import MODEL_NAME, ADAPTER_PATH, CACHE_DIR

_model = None
_tokenizer = None


def load_testmate_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    
    print(f"📦 Loading TestMate (Alternative Method)...")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        local_files_only=True, 
        trust_remote_code=True
    )
    
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    # ✅ ALTERNATIVE FIX: Load without quantization first
    print("🔧 Loading base model (fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print("🔧 Loading LoRA adapter...")
    
    # Load adapter config first
    try:
        adapter_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        print(f"   Adapter type: {adapter_config.peft_type}")
    except Exception as e:
        print(f"⚠️  Could not load adapter config: {e}")
    
    # Load adapter
    _model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False
    )
    
    _model.eval()
    
    # Check device
    device = next(_model.parameters()).device
    print(f"✅ Model loaded on device: {device}")
    
    return _model, _tokenizer


def run_testmate(prompt: str) -> str:
    """Run TestMate model"""
    try:
        model, tokenizer = load_testmate_model()
        
        messages = [
            {"role": "system", "content": "Expert APR agent. Fix code."},
            {"role": "user", "content": prompt}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Get model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        if len(outputs) == 0 or len(outputs[0]) <= inputs['input_ids'].shape[1]:
            return "# Error: No output generated"
        
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:]
        fixed_code = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return fixed_code.replace("```python", "").replace("```", "")
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return f"# Error: {e}"


def run_llamacpp(prompt: str) -> str:
    return run_testmate(prompt)


if __name__ == "__main__":
    print("Testing alternative approach...")
    result = run_testmate("Fix: def divide(a,b): return a*b")
    print(f"Result: {result}")