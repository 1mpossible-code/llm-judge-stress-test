"""Mock backend for testing and offline development"""

import hashlib
import random

from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class MockBackend:
    """Deterministic mock backend."""
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Generate deterministic mock judgment."""
        # Use hash of text + model for determinism
        seed_str = f"{model_spec.name}:{req.text}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        
        rng = random.Random(seed)
        
        # Deterministic label selection
        label = rng.choice(req.allowed_labels)
        
        # Deterministic confidence (0.5 to 0.95)
        confidence = 0.5 + (rng.random() * 0.45)
        
        # Simple rationale
        rationale = f"Mock judgment based on text content and model preferences."
        
        raw_output = f"LABEL: {label}\nCONFIDENCE: {confidence:.2f}\nRATIONALE: {rationale}"
        
        return JudgeResponse(
            raw_output=raw_output,
            usage={"input_tokens": len(req.text) // 4, "output_tokens": 50},
            status="ok",
        )

