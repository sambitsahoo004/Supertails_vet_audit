# revised_scoring_system.py

import json
import re
from typing import Dict, List, Any, Optional
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv

load_dotenv()
# Import the parameters from config.py
from config import PARAMETERS_CONFIG

class CallScorer:
    def __init__(self,
                 api_key: str,
                 model: str = "claude-3-5-sonnet-latest",
                 temperature: float = 0.2):
        """Initialize with Anthropic API key."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        # Load parameters from config
        self.parameters_config = PARAMETERS_CONFIG['parameters']
        
        # Create a mapping for call introduction parameters
        self.call_intro_params = [
            param for param in self.parameters_config 
            if param['name'] in [
                "Self Introduction", 
                "Company Introduction", 
                "Customer Name Confirmation", 
                "Order Confirmation", 
                "Purpose Of the Call (Context)"
            ]
        ]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with structured prompt and handle retries."""
        try:
            system_prompt = """You are a veterinary call scoring assistant. 
            Analyze transcripts of veterinary support calls and provide objective scores based on defined criteria.
            Always respond with valid, well-structured JSON that matches the expected schema.
            Be precise, consistent, and fair in your evaluations.
            """

            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=4000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise

    def get_scoring_prompt_from_config(self, transcript: str) -> str:
        """Generate scoring prompt based on config parameters."""
        prompt_parts = [
            f"""
            Carefully analyze the introduction section of this veterinary support call transcript:

            TRANSCRIPT:
            {transcript}

            Score the following introduction parameters based on their specific criteria:
            """
        ]
        
        # Add scoring criteria for each parameter
        for param in self.call_intro_params:
            param_name = param['name']
            max_score = param['max_score']
            
            prompt_parts.append(f"\n{param_name} (Max Score: {max_score}):")
            
            # Add subparameter details if available
            for subparam in param.get('subparameters', []):
                subparam_name = subparam['name']
                weight = subparam['weight']
                prompt_parts.append(f"- {subparam_name} (Weight: {weight})")
            
            # Add scoring scale based on max_score
            prompt_parts.append(f"Score on a scale of 0-{max_score}, where:")
            prompt_parts.append(f"- 0: Missing completely")
            mid_point = max_score / 2
            prompt_parts.append(f"- {mid_point:.1f}: Partially adequate")
            prompt_parts.append(f"- {max_score}: Perfectly executed")
            prompt_parts.append("")
        
        # Add JSON response format instructions with strengths, shortcomings and improvements
        prompt_parts.append("""
        For each parameter, provide:
        1. Clear strengths with specific evidence from the transcript (max 2-3 bullet points)
        2. Shortcomings as a SINGLE specific sentence that identifies what was missing/inadequate with direct reference to what should have been said or done instead (maximum 30 words)
        3. Actionable improvement of one line which would be short, crisp and impactful by summarizing and analyzing all the shortcomings. Avoid giving examples and use professional English.
        
        CRITICAL FOR SHORTCOMINGS: Instead of generic statements like "Could have been better", provide specific shortcomings that:
        - Quote or reference specific parts of the transcript where the issue occurred
        - Clearly state what should have been said or done instead
        - Be concrete and actionable, not vague
        
        Examples of GOOD shortcomings:
        - "Failed to mention company name 'Supertails' - should have said 'This is Dr. X from Supertails'"
        - "Didn't confirm customer name upfront - should have asked 'Am I speaking with [Name]?' before proceeding"
        - "Skipped explaining call purpose - should have stated 'I'm calling to verify your pharmacy order'"
        
        Examples of BAD shortcomings (avoid these):
        - "Could have been more professional"
        - "Introduction was inadequate"
        - "Needs improvement"
        
        Keep all points extremely concise.
        
        Format your response as a structured JSON with the following format:
        {
            "call_introduction_scores": {
        """)
        
        # Add expected structure for each parameter with single descriptive shortcoming sentence
        for param in self.call_intro_params:
            param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            prompt_parts.append(f"""        "{param_key}": {{
                    "score": X,
                    "max_score": {param['max_score']},
                    "explanation": "Brief explanation with evidence from transcript",
                    "strengths": ["Specific strength 1", "Specific strength 2"],
                    "shortcomings": "Single descriptive sentence summarizing all issues (max 30 words)",
                    "improvements": ["Actionable, concise improvement recommendation"]
                }},""")
        
        # Close the JSON structure
        prompt_parts.append("""
            },
            "introduction_summary": "Brief 1-2 sentence overall assessment",
            "improvement_summary": "3-5 bullet-point priority improvements, each with one brief example"
        }
        
        IMPORTANT: For each parameter, provide shortcomings as a SINGLE descriptive sentence (not a list) limited to a maximum of 20 words.
        Ensure you provide specific, actionable improvements for each parameter that has shortcomings.
        Keep all points extremely concise.
        Use bullet-style formatting only for strengths lists.
        Ensure your response is a valid JSON object.
        """)
        
        return "".join(prompt_parts)

    def score_call_introduction(self, transcript: str) -> Dict[str, Any]:
        """Score the call introduction section of a transcript."""
        
        # Generate scoring prompt based on config
        scoring_prompt = self.get_scoring_prompt_from_config(transcript)

        # Get scores from LLM
        raw_scores = self._call_llm(scoring_prompt)
        try:
            parsed_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
            # Provide a default response structure if parsing fails
            parsed_scores = {
                "call_introduction_scores": {},
                "introduction_summary": "Failed to analyze introduction due to processing error.",
                "improvement_summary": "Unable to generate improvements due to processing error."
            }
            # Add default scores for each parameter
            for param in self.call_intro_params:
                param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
                parsed_scores["call_introduction_scores"][param_key] = {
                    "score": 0,
                    "max_score": param['max_score'],
                    "explanation": "Error evaluating transcript",
                    "strengths": [],
                    "shortcomings": "Unable to analyze this parameter due to processing error.",
                    "improvements": ["Retry analysis"]
                }

        # Calculate total scores
        parameter_scores = {}
        total_weighted_score = 0
        total_max_score = 0
        
        for param in self.call_intro_params:
            param_name = param['name']
            param_key = param_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            max_score = param['max_score']
            
            # Get the raw score
            param_data = parsed_scores["call_introduction_scores"].get(param_key, {})
            raw_score = param_data.get("score", 0)
            
            total_weighted_score += raw_score
            total_max_score += max_score
            
            # Process improvements as a list of strings
            improvements = param_data.get("improvements", [])
            if isinstance(improvements, str):
                improvements = [improvements]
            elif isinstance(improvements, dict):
                # fallback in case format was not followed exactly
                improvements = [
                    f"{param_name}: {improvements.get('recommendation', 'Provide actionable suggestion')}"
                ]
            
            # Process shortcomings - ensure it's a single string, not a list
            shortcomings = param_data.get("shortcomings", "")
            if isinstance(shortcomings, list):
                # If it's still a list, join into a single sentence
                shortcomings = " ".join(shortcomings)
            
            parameter_scores[param_key] = {
                "raw_score": raw_score,
                "max_score": max_score,
                "explanation": param_data.get("explanation", ""),
                "strengths": param_data.get("strengths", []),
                "shortcomings": shortcomings,  # Now a single string
                "improvements": improvements,
                "percentage": (raw_score / max_score) * 100 if max_score > 0 else 0
            }
        
        # Compile results
        results = {
            "category": "call_introduction",
            "parameter_scores": parameter_scores,
            "total_score": total_weighted_score,
            "max_possible_score": total_max_score,
            "percentage_score": (total_weighted_score / total_max_score) * 100 if total_max_score > 0 else 0,
            "summary": parsed_scores.get("introduction_summary", ""),
            "improvement_summary": parsed_scores.get("improvement_summary", "")
        } 
        return results


    def extract_call_introduction(self, transcript: str, max_words: int = 300) -> str:
        """Extract the introduction portion of the call for focused analysis."""
        # Split the transcript into words
        words = transcript.split()
        
        # Take the first N words (approximation of the introduction)
        introduction_words = words[:min(max_words, len(words))]
        
        return " ".join(introduction_words)
    
    def score_transcript_category1(self, transcript: str) -> Dict[str, Any]:
        """Score Category 1 (Call Introduction) for the full transcript."""
        # Extract introduction section for focused analysis
        introduction = self.extract_call_introduction(transcript)
        
        # Score the introduction
        introduction_scores = self.score_call_introduction(introduction)
        
        return introduction_scores


# Example usage
if __name__ == "__main__":
    # Sample test (would be replaced with actual transcript)
    sample_transcript = """
    File: anusree_s_supertails_in__Pharmacy_OB__320__-1__6389215727__2025-05-04_17-22-58.mp3 (Duration: 1:11)

    Dr. Anushree : Hello, hello, hi good evening.

    Ansh Jaiswal : Hello.

    Dr. Anushree : This is Dr. Anushree calling, am I speaking to Mr. Ansh Jaiswal?

    Ansh Jaiswal : Yes.

    Dr. Anushree : Thanks for the confirmation sir. You had ordered cleaner pet anti-tick and flea shampoo for your pet, right?

    Ansh Jaiswal : Yes.

    Dr. Anushree : There is a verification call for that sir, because it is a pharmacy product.

    Ansh Jaiswal : Yes.

    Ansh Jaiswal : I understand.

    Dr. Anushree : So we had to make a prescription from the company. For that, can you tell me the name of your pet?

    Ansh Jaiswal : Roxy.

    Dr. Anushree : How old is Roxy?

    Ansh Jaiswal : Roxy is one year and six months.

    Dr. Anushree : One year and six months. Is it male or female?

    Ansh Jaiswal : Female.

    Ansh Jaiswal : And which breed is it? Labrador.

    Dr. Anushree : How much body weight will it be now sir?

    Ansh Jaiswal : Body weight...

    Dr. Anushree : Yes, the body weight sir?

    Ansh Jaiswal : Let me check.

    Ansh Jaiswal : Around 25 kilos.

    Dr. Anushree : Okay sir.

    Dr. Anushree : And have you used this shampoo before?

    Ansh Jaiswal : No, this is the first time.

    Dr. Anushree : Okay sir.

    Dr. Anushree : So I'll tell you how to use it.

    Dr. Anushree : Okay?


    ---
    Transcript processed on Sat May 10 10:12:52 2025
    """
    
    # Initialize scorer with API key (would be loaded from secure config)
    scorer = CallScorer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Score the transcript
    results = scorer.score_transcript_category1(sample_transcript)
    
    # Print formatted results
    print(json.dumps(results, indent=2))