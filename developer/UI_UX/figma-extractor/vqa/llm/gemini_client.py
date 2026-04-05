"""
Gemini API wrapper with batching, retry, and token management.

Key design decisions:
- Temperature 0 (deterministic, analytical)
- Structured JSON output enforced via response schema
- Token budget per call: max 4096 output tokens
- Image resize to 1024px wide before sending
- Retry with exponential backoff (3 attempts)
- Batch 5-8 crop pairs per call to minimize API round-trips
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Callable
from PIL import Image
import io

from ..config.llm_config import LLMConfig
from ..vision.image_utils import resize_for_llm, image_to_base64, prepare_for_gemini

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Structured response from Gemini API call."""
    success: bool
    data: Optional[Any] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0
    tokens_used: int = 0
    retry_count: int = 0


@dataclass
class BatchResult:
    """Result from a batched API call."""
    batch_index: int
    responses: List[GeminiResponse] = field(default_factory=list)
    total_latency_ms: float = 0


class GeminiClient:
    """Async client for Gemini API with batching and retry logic."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ):
        """Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        if api_key:
            self.config.api_key = api_key
        
        self._model = None
        self._initialized = False
        self._request_semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        if not self.config.is_configured:
            logger.warning("Gemini API key not configured. Set GEMINI_API_KEY env var.")
    
    async def _ensure_initialized(self):
        """Lazily initialize the Gemini model."""
        if self._initialized:
            return
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            self._initialized = True
            logger.info(f"Initialized Gemini client with model: {self.config.model}")
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    async def analyze_images(
        self,
        images: List[Image.Image],
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[Dict] = None,
    ) -> GeminiResponse:
        """Send images to Gemini for analysis.
        
        Args:
            images: List of PIL Images to analyze
            prompt: User prompt describing what to look for
            system_prompt: Optional system instructions
            response_schema: Optional JSON schema for structured output
        
        Returns:
            GeminiResponse with parsed data or error
        """
        await self._ensure_initialized()
        
        if not self._model:
            return GeminiResponse(
                success=False,
                error="Gemini model not initialized"
            )
        
        start_time = time.time()
        retry_count = 0
        
        # Prepare images for Gemini
        prepared_images = []
        for img in images:
            resized = resize_for_llm(img)
            prepared = prepare_for_gemini(resized)
            prepared_images.append(prepared)
        
        # Build the prompt content
        content = []
        
        # Add system prompt if provided
        if system_prompt:
            content.append(system_prompt + "\n\n")
        
        # Add images
        for i, img_data in enumerate(prepared_images):
            content.append(img_data)
            if len(prepared_images) > 1:
                content.append(f"\n[Image {i + 1}]\n")
        
        # Add user prompt
        content.append("\n" + prompt)
        
        # Retry loop with exponential backoff
        last_error = None
        while retry_count <= self.config.max_retries:
            try:
                async with self._request_semaphore:
                    # Run the blocking API call in a thread pool
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._model.generate_content(content)
                    )
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Parse response
                if response and response.text:
                    try:
                        parsed_data = json.loads(response.text)
                        return GeminiResponse(
                            success=True,
                            data=parsed_data,
                            raw_text=response.text,
                            latency_ms=latency_ms,
                            retry_count=retry_count,
                        )
                    except json.JSONDecodeError:
                        # Try to extract JSON from the response
                        extracted = self._extract_json(response.text)
                        if extracted:
                            return GeminiResponse(
                                success=True,
                                data=extracted,
                                raw_text=response.text,
                                latency_ms=latency_ms,
                                retry_count=retry_count,
                            )
                        return GeminiResponse(
                            success=False,
                            error="Failed to parse JSON response",
                            raw_text=response.text,
                            latency_ms=latency_ms,
                            retry_count=retry_count,
                        )
                else:
                    return GeminiResponse(
                        success=False,
                        error="Empty response from Gemini",
                        latency_ms=latency_ms,
                        retry_count=retry_count,
                    )
            
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                
                if retry_count <= self.config.max_retries:
                    # Exponential backoff
                    wait_time = self.config.retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Gemini API error (attempt {retry_count}/{self.config.max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
        
        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000
        return GeminiResponse(
            success=False,
            error=f"Failed after {self.config.max_retries} retries: {last_error}",
            latency_ms=latency_ms,
            retry_count=retry_count,
        )
    
    async def analyze_crop_pairs(
        self,
        crop_pairs: List[Dict],
        system_prompt: str,
        per_pair_prompt_builder: Callable[[Dict, int], str],
    ) -> List[GeminiResponse]:
        """Analyze multiple crop pairs with batching.
        
        Batches crop pairs into groups and runs them concurrently.
        Each batch creates a single API call with multiple image pairs.
        
        Args:
            crop_pairs: List of crop pair dicts with figma_crop, web_crop, context
            system_prompt: System prompt for the analysis
            per_pair_prompt_builder: Function to build prompt for each pair
        
        Returns:
            List of GeminiResponse objects (one per crop pair)
        """
        if not crop_pairs:
            return []
        
        # Create batches
        batches = self._create_batches(crop_pairs, self.config.batch_size)
        
        # Run batches concurrently
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._analyze_batch(
                batch, batch_idx, system_prompt, per_pair_prompt_builder
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch failed with exception: {result}")
                continue
            all_responses.extend(result.responses)
        
        return all_responses
    
    async def _analyze_batch(
        self,
        batch: List[Dict],
        batch_idx: int,
        system_prompt: str,
        per_pair_prompt_builder: Callable[[Dict, int], str],
    ) -> BatchResult:
        """Analyze a single batch of crop pairs."""
        await self._ensure_initialized()
        
        if not self._model:
            return BatchResult(
                batch_index=batch_idx,
                responses=[
                    GeminiResponse(success=False, error="Model not initialized")
                    for _ in batch
                ]
            )
        
        start_time = time.time()
        
        # Build content with all image pairs
        content = [system_prompt + "\n\n"]
        
        for idx, pair in enumerate(batch):
            figma_img = pair.get("figma_crop")
            web_img = pair.get("web_crop")
            
            if not figma_img or not web_img:
                continue
            
            # Add pair header
            content.append(f"\n--- PAIR {idx + 1} ---\n")
            
            # Add Figma image
            content.append(prepare_for_gemini(resize_for_llm(figma_img)))
            content.append("\n[Figma Design]\n")
            
            # Add Web image
            content.append(prepare_for_gemini(resize_for_llm(web_img)))
            content.append("\n[Web Implementation]\n")
            
            # Add per-pair prompt
            pair_prompt = per_pair_prompt_builder(pair, idx)
            content.append(pair_prompt + "\n")
        
        content.append("\nAnalyze all pairs above and respond with a JSON array containing your analysis for each pair.")
        
        try:
            async with self._request_semaphore:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._model.generate_content(content)
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response and response.text:
                try:
                    parsed_data = json.loads(response.text)
                    
                    # If it's a list, distribute to individual responses
                    if isinstance(parsed_data, list):
                        responses = []
                        for idx, item in enumerate(parsed_data):
                            responses.append(GeminiResponse(
                                success=True,
                                data=item,
                                latency_ms=latency_ms / len(batch),
                            ))
                        # Pad if necessary
                        while len(responses) < len(batch):
                            responses.append(GeminiResponse(
                                success=False,
                                error="No data for this pair in batch response"
                            ))
                        return BatchResult(
                            batch_index=batch_idx,
                            responses=responses,
                            total_latency_ms=latency_ms,
                        )
                    else:
                        # Single object response for entire batch
                        return BatchResult(
                            batch_index=batch_idx,
                            responses=[
                                GeminiResponse(
                                    success=True,
                                    data=parsed_data,
                                    latency_ms=latency_ms,
                                )
                            ] * len(batch),
                            total_latency_ms=latency_ms,
                        )
                
                except json.JSONDecodeError:
                    extracted = self._extract_json(response.text)
                    if extracted and isinstance(extracted, list):
                        responses = [
                            GeminiResponse(success=True, data=item, latency_ms=latency_ms / len(extracted))
                            for item in extracted
                        ]
                        while len(responses) < len(batch):
                            responses.append(GeminiResponse(success=False, error="Parse error"))
                        return BatchResult(batch_index=batch_idx, responses=responses, total_latency_ms=latency_ms)
                    
                    return BatchResult(
                        batch_index=batch_idx,
                        responses=[
                            GeminiResponse(
                                success=False,
                                error="JSON parse error",
                                raw_text=response.text,
                            )
                            for _ in batch
                        ],
                        total_latency_ms=latency_ms,
                    )
            else:
                return BatchResult(
                    batch_index=batch_idx,
                    responses=[
                        GeminiResponse(success=False, error="Empty response")
                        for _ in batch
                    ],
                    total_latency_ms=latency_ms,
                )
        
        except Exception as e:
            logger.error(f"Batch {batch_idx} failed: {e}")
            return BatchResult(
                batch_index=batch_idx,
                responses=[
                    GeminiResponse(success=False, error=str(e))
                    for _ in batch
                ],
            )
    
    async def blind_analysis(
        self,
        figma_screenshot: Image.Image,
        web_screenshot: Image.Image,
        system_prompt: str,
    ) -> GeminiResponse:
        """Pass A: Blind visual diff with full screenshots.
        
        Sends both full screenshots with NO DOM hints.
        The model identifies all visual differences it can see.
        
        Args:
            figma_screenshot: Full Figma design screenshot
            web_screenshot: Full web implementation screenshot
            system_prompt: Prompt template for blind analysis
        
        Returns:
            GeminiResponse with list of visual differences found
        """
        prompt = (
            "Image 1 is the FIGMA DESIGN (source of truth).\n"
            "Image 2 is the WEB IMPLEMENTATION (what was built).\n\n"
            "Identify ALL visual differences between these two images."
        )
        
        return await self.analyze_images(
            images=[figma_screenshot, web_screenshot],
            prompt=prompt,
            system_prompt=system_prompt,
        )
    
    async def targeted_analysis(
        self,
        figma_crop: Image.Image,
        web_crop: Image.Image,
        reported_diff: Dict,
        system_prompt: str,
    ) -> GeminiResponse:
        """Pass B: Targeted validation of a specific reported difference.
        
        Args:
            figma_crop: Cropped Figma region
            web_crop: Cropped web region
            reported_diff: DOM diff data for context
            system_prompt: Prompt template for targeted validation
        
        Returns:
            GeminiResponse with validation verdict
        """
        diff_context = (
            f"Reported difference: {reported_diff.get('diff_type', 'unknown')}\n"
            f"Figma value: {reported_diff.get('figma_value')}\n"
            f"Web value: {reported_diff.get('web_value')}\n"
            f"Element: {reported_diff.get('element_name', 'unknown')}"
        )
        
        prompt = (
            f"{diff_context}\n\n"
            "Validate whether this reported difference is:\n"
            "1. CONFIRMED: You can clearly see the difference in the images\n"
            "2. REJECTED: The difference exists in code but is not visible\n"
            "3. UNCERTAIN: Cannot definitively confirm or reject"
        )
        
        return await self.analyze_images(
            images=[figma_crop, web_crop],
            prompt=prompt,
            system_prompt=system_prompt,
        )
    
    def _create_batches(self, items: List, batch_size: int) -> List[List]:
        """Split items into batches of specified size."""
        return [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """Try to extract JSON from a text response.
        
        Handles cases where the model wraps JSON in markdown code blocks
        or adds extra text before/after.
        """
        if not text:
            return None
        
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in code blocks
        import re
        
        # Match ```json...``` or ```...```
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_block_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find array or object bounds
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            start_idx = text.find(start_char)
            end_idx = text.rfind(end_char)
            
            if start_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(text[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics for monitoring."""
        return {
            "model": self.config.model,
            "initialized": self._initialized,
            "is_configured": self.config.is_configured,
            "concurrent_requests": self.config.concurrent_requests,
            "batch_size": self.config.batch_size,
            "max_retries": self.config.max_retries,
        }


# Convenience function for simple usage
async def create_client(api_key: Optional[str] = None) -> GeminiClient:
    """Create and initialize a Gemini client.
    
    Args:
        api_key: Optional API key (or use GEMINI_API_KEY env var)
    
    Returns:
        Initialized GeminiClient
    """
    client = GeminiClient(api_key=api_key)
    await client._ensure_initialized()
    return client
