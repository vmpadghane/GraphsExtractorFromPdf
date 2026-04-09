"""
Ollama Reasoner Module

Uses Ollama LLM for caption validation and graph classification.
"""

import logging
from typing import Dict, Any, List
import base64
from io import BytesIO
from PIL import Image
import ollama


class OllamaReasoner:
    """Handles LLM reasoning for caption selection and graph validation."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = config['ollama']['model']

    def reason_caption(self, graph_image: Image.Image, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to select best caption and validate graph.

        Args:
            graph_image: PIL image of the graph
            candidates: List of candidate captions

        Returns:
            Dict with validation results
        """
        self.logger.debug("Reasoning about caption and graph type")

        # Convert image to base64
        buffered = BytesIO()
        graph_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare candidates text
        candidates_text = "\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(candidates)])

        prompt = f"""
Analyze this image and the provided text candidates.

First, determine if this image is actually a graph/chart/plot. If not, respond with "NOT_GRAPH".

If it is a graph, classify the type: line graph, bar chart, scatter plot, mathematical plot, coordinate geometry figure, or other.

Then, select the best caption from the candidates that describes this graph. Choose the most relevant one, or "NONE" if none fit.

Candidates:
{candidates_text}

Respond in JSON format:
{{
    "is_graph": true/false,
    "graph_type": "type or null",
    "caption": "selected caption text or null",
    "confidence": 0.0-1.0
}}
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_str]
                }],
                options={
                    'temperature': self.config['ollama']['temperature'],
                    'num_predict': self.config['ollama']['max_tokens']
                }
            )

            # Parse response (assuming JSON output)
            import json
            result = json.loads(response['message']['content'])

            self.logger.debug(f"LLM result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in LLM reasoning: {e}")
            return {
                "is_graph": False,
                "graph_type": None,
                "caption": None,
                "confidence": 0.0
            }