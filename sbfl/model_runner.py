import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from inference import load_model

_model = None
_tokenizer = None

def load_testmate_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    print("📦 Loading TestMate...")
    _model, _tokenizer = load_model()
    print("✅ Model loaded successfully")
    return _model, _tokenizer

def run_testmate(prompt: str) -> str:
    try:
        model, tokenizer = load_testmate_model()
        
        messages = [
            {"role": "system", "content": "Expert APR agent. Fix code using stack trace."},
            {"role": "user", "content": prompt}
        ]
        
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, 
                pad_token_id=tokenizer.pad_token_id
            )
        
        fixed_code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return fixed_code
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""