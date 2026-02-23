import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_NAME, OUTPUT_DIR, CACHE_DIR, ADAPTER_PATH

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("Loading Base Model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},  
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    
    print(f"Loading Peft Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH) 
    return model, tokenizer

def repair_code(issue, buggy_code, model, tokenizer):
    messages = [
        {"role": "system", "content": "Expert APR agent. Fix code using stack trace."},
        {"role": "user", "content": f"ISSUE: {issue}\n\nCODE:\n{buggy_code}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.1, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)