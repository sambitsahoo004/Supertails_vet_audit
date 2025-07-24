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

class PetInformationScorer:
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
        
        # Create a mapping for pet information parameters
        self.pet_info_params = [
            param for param in self.parameters_config 
            if param['name'] in [
                "Pet Name, Age and Gender", 
                "Pet Body Weight", 
                "Health Concern"
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

    def get_pet_info_scoring_prompt(self, transcript: str) -> str:
        """Generate scoring prompt for pet information parameters."""
        prompt_parts = [
            f"""
            Carefully analyze this veterinary support call transcript for information about the pet:

            TRANSCRIPT:
            {transcript}

            Score the following pet information parameters based on their specific criteria:
            """
        ]
        
        # Add scoring criteria for each parameter
        for param in self.pet_info_params:
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
        
        # Add JSON response format instructions with descriptive shortcomings
        prompt_parts.append("""
        For each parameter, provide:
        1. Clear strengths with specific evidence from the transcript (max 2-3 bullet points)
        2. Shortcomings as a SINGLE specific sentence that identifies what was missing/inadequate with direct reference to what should have been asked or obtained instead (maximum 30 words)
        3. Actionable improvement of one line which would be short, crisp and impactful by summarizing and analyzing all the shortcomings. Avoid giving examples and use professional English.
        
        CRITICAL FOR SHORTCOMINGS: Instead of generic statements like "Could have gathered more information", provide specific shortcomings that:
        - Quote or reference specific parts of the transcript where information was missing or inadequate
        - Clearly state what specific information should have been obtained
        - Be concrete and actionable, not vague
        
        Examples of GOOD shortcomings:
        - "Asked 'How old is Roxy?' but didn't clarify exact birth date for precise medication dosing"
        - "Obtained weight '25 kilos' but failed to ask when pet was last weighed for accuracy"
        - "Didn't inquire about current health issues despite ordering tick/flea treatment - should have asked about existing symptoms"
        
        Examples of BAD shortcomings (avoid these):
        - "Could have gathered more information"
        - "Pet information was incomplete"
        - "Needs better data collection"
        
        Keep all points extremely concise.

        Format your response as a structured JSON with the following format:
        {
            "pet_information_scores": {
        """)
        
        # Add expected structure for each parameter with descriptive shortcomings
        for param in self.pet_info_params:
            param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            prompt_parts.append(f"""        "{param_key}": {{
                    "score": X,
                    "max_score": {param['max_score']},
                    "explanation": "Brief one-sentence explanation",
                    "strengths": ["Concise strength 1", "Concise strength 2"],
                    "shortcomings": "Single descriptive sentence summarizing all issues (max 20 words)",
                    "improvements": ["Actionable, concise improvement recommendation"]
                }},""")
        
        # Close the JSON structure
        prompt_parts.append("""
            },
            "pet_information_summary": "Brief 1-2 sentence overall assessment",
            "improvement_summary": "3-5 bullet-point priority improvements, each with one brief example"
        }
        
        IMPORTANT: For each parameter, provide shortcomings as a SINGLE descriptive sentence (not a list) limited to a maximum of 30 words.
        Ensure you provide specific, actionable improvements for each parameter that has shortcomings.
        Keep all points extremely concise.
        Use bullet-style formatting only for strengths lists.
        Ensure your response is a valid JSON object.
        """)
        
        return "".join(prompt_parts)

    def score_pet_information(self, transcript: str) -> Dict[str, Any]:
        """Score the pet information gathering section of a transcript."""
        
        # Generate scoring prompt
        scoring_prompt = self.get_pet_info_scoring_prompt(transcript)

        # Get scores from LLM
        raw_scores = self._call_llm(scoring_prompt)
        try:
            parsed_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
            # Provide a default response structure if parsing fails
            parsed_scores = {
                "pet_information_scores": {},
                "pet_information_summary": "Failed to analyze pet information due to processing error.",
                "improvement_summary": "Unable to generate improvements due to processing error."
            }
            # Add default scores for each parameter
            for param in self.pet_info_params:
                param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
                parsed_scores["pet_information_scores"][param_key] = {
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
        
        for param in self.pet_info_params:
            param_name = param['name']
            param_key = param_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            max_score = param['max_score']
            
            # Get the raw score
            param_data = parsed_scores["pet_information_scores"].get(param_key, {})
            raw_score = param_data.get("score", 0)
            
            # Add up scores
            total_weighted_score += raw_score
            total_max_score += max_score
            
            # Process improvements (now expected to be a list of concise strings)
            improvements = param_data.get("improvements", [])
            if isinstance(improvements, str):
                improvements = [improvements]
            elif isinstance(improvements, dict):
                # fallback in case format wasn't followed
                improvements = [f"{param_name}: {improvements.get('recommendation', 'Provide actionable suggestion')}"]

            # Process shortcomings - ensure it's a single string, not a list
            shortcomings = param_data.get("shortcomings", "")
            if isinstance(shortcomings, list):
                # If it's still a list, join into a single sentence
                shortcomings = " ".join(shortcomings)

            # Store the parameter score
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
            "category": "pet_information",
            "parameter_scores": parameter_scores,
            "total_score": total_weighted_score,
            "max_possible_score": total_max_score,
            "percentage_score": (total_weighted_score / total_max_score) * 100 if total_max_score > 0 else 0,
            "summary": parsed_scores.get("pet_information_summary", ""),
            "improvement_summary": parsed_scores.get("improvement_summary", "")
        }
        
        return results


    def extract_relevant_transcript_section(self, transcript: str) -> str:
        """
        Extract the relevant section of the transcript for pet information analysis.
        This function uses patterns in conversation to identify where pet information is discussed.
        """
        # This is a simplified approach - in a real implementation, we might use 
        # more sophisticated text analysis or specific patterns
        
        # Define patterns that might indicate pet information sections
        pet_info_patterns = [
            r'(pet|dog|cat|animal).*?(name|age|gender|weight|health|concern|issue|problem)',
            r'(weight|weigh|pounds|kilos)',
            r'(health|concern|issue|problem|symptom)',
            r'(male|female|boy|girl)'
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(pet_info_patterns)
        
        # Initialize variables for tracking context
        lines = transcript.split('\n')
        relevant_lines = []
        context_window = 4  # Number of lines before and after a matching line to include
        
        # Find relevant sections with context
        for i, line in enumerate(lines):
            if re.search(combined_pattern, line, re.IGNORECASE):
                # Include context lines before
                start_idx = max(0, i - context_window)
                relevant_lines.extend(lines[start_idx:i])
                
                # Include the matching line
                relevant_lines.append(line)
                
                # Include context lines after
                end_idx = min(len(lines), i + context_window + 1)
                relevant_lines.extend(lines[i+1:end_idx])
        
        # Remove duplicates while preserving order
        unique_relevant_lines = []
        for line in relevant_lines:
            if line not in unique_relevant_lines:
                unique_relevant_lines.append(line)
        
        # If we didn't find relevant sections or if the selection is very small,
        # return a larger portion of the transcript
        if len(unique_relevant_lines) < 10:
            # Take a significant chunk of the transcript
            chunk_size = min(len(lines), 100)  # Limit to 100 lines
            return '\n'.join(lines[:chunk_size])
        
        return '\n'.join(unique_relevant_lines)
    
    def score_transcript_category2(self, transcript: str) -> Dict[str, Any]:
        """Score Category 2 (Pet Information) for the full transcript."""
        # Extract relevant section for focused analysis
        relevant_section = self.extract_relevant_transcript_section(transcript)
        
        # Score the pet information
        pet_info_scores = self.score_pet_information(relevant_section)
        
        return pet_info_scores
    
# Example usage
if __name__ == "__main__":
    # Sample test transcript
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
    scorer = PetInformationScorer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Score the transcript and generate detailed feedback
    results = scorer.score_transcript_category2(sample_transcript)
    print(json.dumps(results, indent=2))