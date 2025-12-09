import os
from safetensors import safe_open

def calculate_non_embedding_params(filenames):
    total_params = 0
    skipped_params = 0
    
    # Common names for embedding layers in various architectures (Llama, GPT, BERT)
    # You can add "lm_head" here if you also want to exclude the final output layer.
    EMBEDDING_KEYWORDS = ["embed_tokens", "wte", "word_embeddings", "embeddings"]

    print(f"{'File Name':<40} | {'Active Params':>15}")
    print("-" * 60)

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Error: {filename} not found.")
            continue
            
        file_params = 0
        
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                # 1. Check if this key belongs to an embedding layer
                is_embedding = any(keyword in key for keyword in EMBEDDING_KEYWORDS)
                
                # Get shape info
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                
                # Calculate element count
                params = 1
                for dim in shape:
                    params *= dim
                
                if is_embedding:
                    # Log what we are skipping
                    print(f"   [SKIP] Excluding embedding layer: {key} ({params:,} params)")
                    skipped_params += params
                else:
                    # Add to total if not embedding
                    file_params += params
        
        print(f"{filename:<40} | {file_params:>15,}")
        total_params += file_params

    print("-" * 60)
    print(f"{'TOTAL NON-EMBEDDING PARAMS':<40} | {total_params:>15,}")
    print(f"{'SKIPPED EMBEDDING PARAMS':<40} | {skipped_params:>15,}")
    
    # Convert to billions
    print(f"\nNet Model Size: {total_params / 1e9:.2f} Billion parameters")

def calculate_parameters(filenames):
    total_params = 0
    
    print(f"{'File Name':<40} | {'File Params':>15}")
    print("-" * 60)

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Error: {filename} not found.")
            continue
            
        file_params = 0
        
        # Open the safetensors file securely
        with safe_open(filename, framework="pt", device="cpu") as f:
            # Iterate over all keys (tensor names) in the file
            for key in f.keys():
                # Get the tensor slice/info without loading the full tensor into RAM
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                
                # Calculate elements in this tensor (e.g., [2, 3] -> 6 parameters)
                params = 1
                for dim in shape:
                    params *= dim
                
                file_params += params
        
        print(f"{filename:<40} | {file_params:>15,}")
        total_params += file_params

    print("-" * 60)
    print(f"{'TOTAL PARAMETERS':<40} | {total_params:>15,}")
    
    # Convert to billions for easier reading
    print(f"\nApproximate Model Size: {total_params / 1e9:.2f} Billion parameters")

# List your specific checkpoint files here
checkpoint_files = [
    # "OLMo2-1B-stage2-seed42-SEXMH-L5/step23852-unsharded/model-00001-of-00002.safetensors",
    # "OLMo2-1B-stage2-seed42-SEXMH-L5/step23852-unsharded/model-00002-of-00002.safetensors"
    "OLMo2-1B-stage2-seed42-NONE/step23852-unsharded/model-00001-of-00002.safetensors",
    "OLMo2-1B-stage2-seed42-NONE/step23852-unsharded/model-00002-of-00002.safetensors"
]

if __name__ == "__main__":
    calculate_non_embedding_params(checkpoint_files)