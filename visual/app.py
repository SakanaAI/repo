from flask import Flask, render_template, request, jsonify
import numpy as np
import traceback
import torch
import sys
import queue
import threading
from concurrent.futures import Future
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import OrderedDict

app = Flask(__name__)
CKPT_NAME = "SakanaAI/RePo-OLMo2-1B-stage2-L5"

# --- 1. SETUP QUEUE & LOCKING ---
# We use a Queue to serialize requests so the GPU is only accessed by one thread at a time.
execution_queue = queue.Queue()

class RePo:
    def __init__(self, model_name="SakanaAI/RePo-OLMo2-1B-stage2-L5", start_layer=5):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda") 
            self.device = torch.device("cuda")
            print("Model loaded on CUDA.")
        else:
            print("[Warning] No GPU available, the service may be super slow")
            self.device = torch.device("cpu")
        self.model = model
        self.start_layer = start_layer
        self.cache = OrderedDict()
        self.cache_size = 8
    
    @torch.no_grad()
    def forward(self, prompt, layer, head, max_tokens=512):
        truncated = False
        inputs = self.tokenizer(prompt, return_tensors="pt")
        seq_len = inputs['input_ids'].shape[1]

        if seq_len > max_tokens:
            truncated = True
            inputs['input_ids'] = inputs['input_ids'][:, :max_tokens]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_tokens]
            prompt = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)

        if prompt in self.cache:
            pred_indices, toks = self.cache[prompt]
            self.cache.move_to_end(prompt)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            tok_ids = inputs['input_ids']
            toks = self.tokenizer.convert_ids_to_tokens(tok_ids.squeeze(0), skip_special_tokens=False)
            toks = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in toks]
            n_toks = len(toks)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True, output_pred_indices=True)
            pred_indices = outputs.pred_indices 
            pred_indices = [it.data.squeeze(0).reshape(-1, n_toks).tolist() for it in pred_indices]
            self.cache[prompt] = (pred_indices, toks)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        
        data = []
        # Safety check for layer bounds
        if layer < len(pred_indices):
            for x, (y, t) in enumerate(zip(pred_indices[layer][head], toks)):
                data.append({
                    "x": int(x),
                    "y": float(y),
                    "t": str(t)
                })
        return data, truncated

# Initialize model globally
model = RePo(CKPT_NAME)

# --- 2. BACKGROUND WORKER ---
def worker():
    """
    Consumer thread that processes requests sequentially.
    """
    print("Background worker started.")
    while True:
        # Get a job from the queue
        # job structure: (future_object, args_dict)
        future, args = execution_queue.get()
        try:
            # Run the heavy model inference
            result = model.forward(
                prompt=args['sentence'], 
                layer=args['layer'], 
                head=args['head'], 
                max_tokens=args['max_tokens']
            )
            # Pass result back to the waiting HTTP thread
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            execution_queue.task_done()

# Start the worker thread
threading.Thread(target=worker, daemon=True).start()


@app.route('/')
def index():
    return render_template('index.html')

# --- 3. NEW STATUS ENDPOINT ---
@app.route('/queue_status', methods=['GET'])
def queue_status():
    """Returns the current number of requests waiting in queue."""
    return jsonify({"count": execution_queue.qsize()})

@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    try:
        req_data = request.json
        sentence = req_data.get('sentence', '')
        layer = int(req_data.get('layer', 5))
        head = int(req_data.get('head', 0))
        
        future = Future()
        execution_queue.put((future, {
            'sentence': sentence,
            'layer': layer,
            'head': head,
            'max_tokens': 512
        }))
        
        results, was_truncated = future.result()

        # --- OUTLIER DETECTION LOGIC ---
        suggested_range = None
        y_vals = [d['y'] for d in results]
        
        # Only apply logic if we have enough data points
        if len(y_vals) > 5:
            # Calculate Quartiles
            q75, q25 = np.percentile(y_vals, [75 ,25])
            iqr = q75 - q25
            
            # Define bounds (1.5 * IQR is standard for outliers)
            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)
            
            # Find the actual data range within these bounds
            inliers = [y for y in y_vals if lower_bound <= y <= upper_bound]
            
            if inliers:
                # Add 5% padding for visual comfort
                min_in = min(inliers)
                max_in = max(inliers)
                padding = (max_in - min_in) * 0.05
                if padding == 0: padding = 1.0 # Handle flat lines
                
                suggested_range = [min_in - padding, max_in + padding]

        return jsonify({
            "status": "success", 
            "data": results, 
            "truncated": was_truncated,
            "suggested_range": suggested_range
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# Note: The if __name__ == '__main__' block is handled by uvicorn in your docker command