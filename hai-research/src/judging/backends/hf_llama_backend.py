"""HuggingFace local Llama backend"""

import os
import torch
from typing import Any

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


# Global model cache
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)


class HuggingFaceBackend:
    """HuggingFace transformers backend for local models."""
    
    def __init__(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package not installed. Install with: pip install -e .[hf]"
            )
    
    def _get_model(self, model_spec: ModelSpec):
        """Get or load model and tokenizer (cached)."""
        model_name = model_spec.name
        
        if model_name in _model_cache:
            return _model_cache[model_name]
        
        device = model_spec.params.get("device", "cpu")
        dtype_str = model_spec.params.get("dtype", "float16")
        
        # Check if CUDA is actually available
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.float16)
        
        # Use float32 for CPU if bfloat16 requested (CPU doesn't support bfloat16)
        if device == "cpu" and dtype == torch.bfloat16:
            dtype = torch.float32
            print(f"Warning: bfloat16 not supported on CPU, using float32 instead")
        
        print(f"Loading model {model_name} (this may take a while on first run)...")
        print(f"  Device: {device}, Dtype: {dtype_str}")
        
        # Get HuggingFace token for gated models (optional)
        hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token:
            print(f"  Using HuggingFace authentication token")
        else:
            print(f"  Warning: No HUGGINGFACE_HUB_TOKEN found - gated models may fail")
        
        try:
            # Pass token to from_pretrained for gated models
            tokenizer_kwargs = {}
            model_kwargs = {}
            if hf_token:
                tokenizer_kwargs["token"] = hf_token
                model_kwargs["token"] = hf_token
            
            tokenizer = self.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            print(f"Tokenizer loaded successfully")
            
            # Load model with appropriate device handling
            if device == "cpu":
                # For CPU, load directly to CPU (no device_map)
                model = self.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    **model_kwargs,
                )
                model = model.to("cpu")
            else:
                # For CUDA, use device_map="auto" or "cuda"
                model = self.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto",  # Let transformers handle device placement
                    **model_kwargs,
                )
            
            print(f"Model loaded successfully")
            model.eval()
            
            _model_cache[model_name] = (model, tokenizer)
            return model, tokenizer
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            # Provide helpful error message for common issues
            if "does not appear to have a file named" in str(e):
                print(f"\nPossible solutions:")
                print(f"1. Check if model ID is correct (e.g., 'meta-llama/Llama-2-7b-chat-hf' instead of 'meta-llama/Llama-2-7b-chat')")
                print(f"2. Verify you have access to the model on HuggingFace: https://huggingface.co/{model_name}")
                print(f"3. Ensure HUGGINGFACE_HUB_TOKEN is set and has access to gated models")
                print(f"4. Try using huggingface-cli login to authenticate")
            
            raise RuntimeError(error_msg) from e
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Generate judgment using local model."""
        import torch
        
        try:
            model, tokenizer = self._get_model(model_spec)
            
            # Format prompt (use chat template if available)
            # Store original prompt for later extraction
            original_prompt = req.prompt
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": req.prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback formatting
                formatted_prompt = req.prompt
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with deterministic-ish seed
            max_new_tokens = model_spec.params.get("max_new_tokens", req.max_tokens)
            
            # Set seed for reproducibility (based on prompt hash for determinism)
            import hashlib
            seed = int(hashlib.md5(req.text.encode()).hexdigest()[:8], 16) % (2**31)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            with torch.no_grad():
                # For greedy decoding (temperature=0), don't use do_sample
                # For sampling, use temperature
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                if req.temperature > 0:
                    gen_kwargs["temperature"] = req.temperature
                    gen_kwargs["do_sample"] = True
                else:
                    # Greedy decoding - don't pass temperature
                    gen_kwargs["do_sample"] = False
                
                outputs = model.generate(**gen_kwargs)
            
            # Decode only new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # If still empty, try decoding full output and removing input
            if not raw_output.strip():
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Try to remove the input prompt from the full output
                # Handle chat template formatting (may include special tokens)
                if formatted_prompt.strip() in full_output:
                    raw_output = full_output.replace(formatted_prompt.strip(), "", 1).strip()
                elif original_prompt.strip() in full_output:
                    raw_output = full_output.replace(original_prompt.strip(), "", 1).strip()
                else:
                    # For chat templates, extract only the assistant response
                    # Llama chat templates often format as: <user>...<assistant>...
                    if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
                        parts = full_output.split("<|start_header_id|>assistant<|end_header_id|>")
                        if len(parts) > 1:
                            raw_output = parts[-1].strip()
                            # Remove any trailing special tokens
                            raw_output = raw_output.replace("<|eot_id|>", "").strip()
                    else:
                        # Fallback: use token-based extraction
                        raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            return JudgeResponse(
                raw_output=raw_output,
                usage=None,  # Local models don't provide usage
                status="ok",
            )
        
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            import traceback
            print(f"ERROR in HuggingFace backend: {error_msg}")
            print(traceback.format_exc())
            return JudgeResponse(
                raw_output="",
                usage=None,
                status="error",
                error=error_msg,
            )

