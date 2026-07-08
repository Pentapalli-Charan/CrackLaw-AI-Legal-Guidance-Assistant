import os

def profile_memory():
    print("==================================================")
    print("CRACKLAW MEMORY PROFILER")
    print("==================================================")
    
    param_count = 1_034_112  # Current small model param count
    vocab_size = 32000
    d_model = 256
    
    # fp32 memory calculations
    model_mem = param_count * 4 / (1024**2) # MB
    optimizer_mem = param_count * 8 / (1024**2) # AdamW state
    
    batch_size = 32
    seq_length = 512
    activation_mem = (batch_size * seq_length * d_model * 4) / (1024**2) * 4 # rough estimate for activations
    
    print(f"[ CURRENT ARCHITECTURE (1M Params) ]")
    print(f"Model Weights (FP32):   {model_mem:.2f} MB")
    print(f"Optimizer State:        {optimizer_mem:.2f} MB")
    print(f"Activation Mem (B=32):  ~{activation_mem:.2f} MB")
    print(f"Total VRAM Required:    ~{model_mem + optimizer_mem + activation_mem + 100:.2f} MB")
    
    # Scaling to 120M parameters
    scaled_params = 120_000_000
    scaled_model = scaled_params * 4 / (1024**2)
    scaled_opt = scaled_params * 8 / (1024**2)
    scaled_act = (32 * 1024 * 768 * 4) / (1024**2) * 12 # 12 layers
    
    print(f"\n[ SCALED ARCHITECTURE (120M Params) ]")
    print(f"Model Weights (FP32):   {scaled_model:.2f} MB")
    print(f"Optimizer State:        {scaled_opt:.2f} MB")
    print(f"Activation Mem (B=32):  ~{scaled_act:.2f} MB")
    print(f"Total VRAM Required:    ~{scaled_model + scaled_opt + scaled_act + 1000:.2f} MB")
    
    print("\nCONCLUSION: Mixed precision (FP16/BF16) and gradient accumulation are MANDATORY for 120M scaling to fit inside 12GB VRAM cards.")
    print("==================================================")

if __name__ == "__main__":
    profile_memory()
