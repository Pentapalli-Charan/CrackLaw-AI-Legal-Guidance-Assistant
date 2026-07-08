import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.training.loss import LanguageModelingLoss

def verify_gradients():
    print("==================================================")
    print("CRACKLAW GRADIENT PROPAGATION & ARCHITECTURE VERIFIER")
    print("==================================================")
    
    # Initialize a tiny model
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_len=64
    )
    model = CrackLawTransformer(config)
    
    # Dummy inputs
    batch_size = 2
    seq_len = 16
    src_input_ids = torch.randint(0, 100, (batch_size, seq_len))
    tgt_input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    logits = model(src_input_ids, tgt_input_ids)
    
    # Dummy loss
    labels = torch.randint(0, 100, (batch_size, seq_len))
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 100), labels.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    dead_parameters = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            if param.grad is None:
                dead_parameters.append((name, "No Gradient (None)"))
            elif torch.all(param.grad == 0):
                dead_parameters.append((name, "Zero Gradient"))
                
    print(f"Total Parameter Tensors Checked: {total_params}")
    print(f"Dead Parameters Detected:        {len(dead_parameters)}")
    
    if len(dead_parameters) > 0:
        print("\n[ ERROR: DEAD PARAMETERS FOUND ]")
        for name, reason in dead_parameters:
            print(f"  - {name}: {reason}")
    else:
        print("\n[ SUCCESS: 100% GRADIENT PROPAGATION ]")
        print("Gradients successfully flowed from the LM Head, through the Decoder stack, Cross-Attention, Encoder stack, down to the Source Embeddings.")
        
    print("==================================================")

if __name__ == "__main__":
    verify_gradients()
