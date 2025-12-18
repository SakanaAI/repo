from flask import Flask, render_template, request, jsonify
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

app = Flask(__name__)
CKPT_NAME="SakanaAI/RePo-OLMo2-1B-stage2-L5"

class RePo:
    def __init__(self, model_name="SakanaAI/RePo-OLMo2-1B-stage2-L5", start_layer=5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda") 
            self.device = torch.device("cuda")
        else:
            print("[Warning] No GPU available, the service may be super slow")
            self.device = torch.device("cpu")
        self.model = model
        self.start_layer = start_layer
        self.prev = None
        self.prev_indices = None
        self.prev_tok = None
    
    @torch.no_grad()
    def forward(self, prompt, layer, head):
        if self.prev == prompt:
            pred_indices = self.prev_indices
            toks = self.prev_tok
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            tok_ids = inputs['input_ids']
            toks = self.tokenizer.convert_ids_to_tokens(tok_ids.squeeze(0), skip_special_tokens=False)
            n_toks = len(toks)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True, output_pred_indices=True)
            pred_indices = outputs.pred_indices # [L, B, H * N]
            pred_indices = [it.data.squeeze(0).reshape(-1, n_toks).tolist() for it in pred_indices] # [L, H, N]
            self.prev = prompt
            self.prev_indices = pred_indices
            self.prev_tok = toks
        data = []
        for x, (y, t) in enumerate(zip(pred_indices[layer][head], toks)):
            data.append({
                "x": int(x),
                "y": float(y),
                "t": str(t)
            })
        return data

model = RePo(CKPT_NAME)

def _test_repo():
    sentence = "The quick brown fox jumps over the lazy dog. "
    layer = 5
    head = 1
    data = model.forward(sentence, layer, head)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    req_data = request.json
    sentence = req_data.get('sentence', '')
    layer = int(req_data.get('layer', 5))
    head = int(req_data.get('head', 0))

    # Calculate data
    results = model.forward(sentence, layer, head)
    return jsonify({"status": "success", "data": results})

if __name__ == '__main__':
    _test_repo()
    app.run(debug=True, port=5000)
