"""Rule-based scorer of ASTERISM.

No LLM judge is used, and answer extraction never looks at the gold answer:
output -> final answer block -> typed answer for the item's answer format -> 0/1 against the gold answer.
An output without a valid answer scores 0.
"""
from .judge import INVALID, judge
from .score import contract_of, score_generations, score_output

__all__ = ["INVALID", "judge", "contract_of", "score_output", "score_generations"]
