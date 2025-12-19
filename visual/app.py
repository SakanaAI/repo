from flask import Flask, render_template, request, jsonify
import numpy as np
import traceback
import torch
import sys
import queue
import threading
from concurrent.futures import Future
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


app = Flask(__name__)
CKPT_NAME="SakanaAI/RePo-OLMo2-1B-stage2-L5"

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
        self.prev = None
        self.prev_indices = None
        self.prev_tok = None
    
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

        if self.prev == prompt:
            pred_indices = self.prev_indices
            toks = self.prev_tok
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
            self.prev = prompt
            self.prev_indices = pred_indices
            self.prev_tok = toks
        
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
        
        # Create a Future object to communicate between threads
        future = Future()
        
        # Push job to queue
        execution_queue.put((future, {
            'sentence': sentence,
            'layer': layer,
            'head': head,
            'max_tokens': 512
        }))
        
        # Wait for the result (This blocks the HTTP request until the worker finishes)
        # We can add a timeout here if desired (e.g., future.result(timeout=60))
        results, was_truncated = future.result()

        return jsonify({
            "status": "success", 
            "data": results, 
            "truncated": was_truncated
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

# Note: The if __name__ == '__main__' block is handled by uvicorn in your docker command