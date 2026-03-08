from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ast

app = FastAPI(title="TestMate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_features(source_code: str):
    """
    Parses the Python source code and extracts key project features
    like classes, functions, and their parameters using the AST.
    """
    features = {
        "classes": [],
        "functions": [],
        "total_lines": len(source_code.splitlines())
    }
    
    try:
        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            # Extract Class information
            if isinstance(node, ast.ClassDef):
                features["classes"].append({
                    "name": node.name,
                    "line_number": node.lineno
                })
            
            # Extract Function information
            elif isinstance(node, ast.FunctionDef):
                # Get argument names
                args = [arg.arg for arg in node.args.args]
                features["functions"].append({
                    "name": node.name,
                    "arguments": args,
                    "line_number": node.lineno,
                    "returns": ast.unparse(node.returns) if node.returns else None
                })
                
        return features
    except SyntaxError as e:
        return {"error": f"Failed to parse code: Invalid Python syntax at line {e.lineno}"}

@app.post("/api/repair")
async def run_feature_extraction(file: UploadFile = File(...)):
    # 1. Read and decode the uploaded Python file
    content = await file.read()
    source_code = content.decode("utf-8")
    
    print(f"📥 Received file: {file.filename}")
    
    # 2. Run the Feature Extraction step
    extracted_data = extract_features(source_code)
    
    # Check if there was a syntax error during extraction
    if "error" in extracted_data:
         return {
            "status": "error",
            "message": extracted_data["error"]
        }

    print("⚙️ Extracted Features:", extracted_data)
    
    # 3. Return the success response back to React
    # You can later pass this extracted_data to your test generation prompt
    return {
        "status": "success",
        "filename": file.filename,
        "message": f"Extracted {len(extracted_data['classes'])} classes and {len(extracted_data['functions'])} functions.",
        "features": extracted_data
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)