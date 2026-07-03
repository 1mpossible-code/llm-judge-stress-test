"""Base backend interface"""

from typing import Protocol
from src.schemas import ModelSpec, JudgeRequest, JudgeResponse


class JudgeBackend(Protocol):
    """Protocol for judge backends."""
    
    def judge(self, model_spec: ModelSpec, req: JudgeRequest) -> JudgeResponse:
        """Execute judgment request.
        
        Args:
            model_spec: Model specification
            req: Judgment request
            
        Returns:
            JudgeResponse with raw output and status
        """
        ...

