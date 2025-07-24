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

class CallConclusionScorer:
    def __init__(self,
                 api_key: str,
                 model: str = "claude-3-5-sonnet-latest",
                 temperature: float = 0.2):
        """Initialize with Anthropic API key."""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        # Extract Category 5 parameters from the config
        self.call_conclusion_params = self._extract_call_conclusion_params()
    
    def _extract_call_conclusion_params(self) -> List[Dict[str, Any]]:
        """Extract Category 5 (Call Conclusion) parameters from the config."""
        # Get all parameters
        all_parameters = PARAMETERS_CONFIG.get("parameters", [])
        
        # Filter for Category 5 parameters (Call Conclusion)
        call_conclusion_parameters = [
            param for param in all_parameters 
            if param.get("name") in [
                "Provided 'Chat with the Vet' Reminder at Call Closing",
                "Previous Prescription",
                "Medicine Usage"
            ]
        ]
        
        return call_conclusion_parameters
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with structured prompt and handle retries."""
        try:
            system_prompt = """You are a veterinary call scoring assistant. 
            Analyze transcripts of veterinary support calls and provide objective scores based on defined criteria.
            Always respond with valid, well-structured JSON that matches the expected schema.
            Be precise, consistent, and fair in your evaluations.
            Focus specifically on how well the call was concluded.
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

    def get_call_conclusion_scoring_prompt(self, transcript: str) -> str:
        """Generate scoring prompt for call conclusion parameters."""
        prompt_parts = [
            f"""
            Carefully analyze this veterinary support call transcript, focusing specifically on how the call was concluded:

            TRANSCRIPT:
            {transcript}

            Score the following call conclusion parameters based on their specific criteria:
            """
        ]
        
        # Add scoring criteria for each parameter
        for param in self.call_conclusion_params:
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
        2. For any parameter not receiving a perfect score, identify the specific shortcoming with:
           - A direct quote or specific moment from the transcript showing the gap
           - What exactly should have been said or done instead
           - FORMAT: "At [specific moment/quote], [problem description], [provide some suggestive example]" (maximum 30 words)
        3. Actionable improvement of one line which would be short, crisp and impactful by summarizing and analyzing all the shortcomings. Use professional English.
        
        Keep all points extremely concise.

        Format your response as a structured JSON with the following format:
        {
            "call_conclusion_scores": {
        """)
        
        # Add expected structure for each parameter - with shortcomings as a string, not an array
        for param in self.call_conclusion_params:
            param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '')
            prompt_parts.append(f"""        "{param_key}": {{
                    "score": X,
                    "max_score": {param['max_score']},
                    "explanation": "Brief explanation with evidence from transcript",
                    "strengths": ["Specific strength 1", "Specific strength 2"],
                    "shortcomings": "Single descriptive paragraph of shortcomings within 30 words total",
                    "improvements": ["Actionable, concise improvement recommendation"]
                }},""")
        
        # Close the JSON structure
        prompt_parts.append("""
            },
            "call_conclusion_summary": "Brief 1-2 sentence overall assessment",
            "improvement_summary": "3-5 bullet-point priority improvements, each with one brief example"
        }
        
        Ensure you provide specific, actionable improvements for each parameter that has shortcomings.
        Keep all points extremely concise.
        Use bullet-style formatting for strengths, but make shortcomings a single descriptive paragraph (max 30 words).
        Ensure your response is a valid JSON object.
        """)
        
        return "".join(prompt_parts)

    def extract_call_conclusion_section(self, transcript: str) -> str:
        """
        Extract the conclusion section of the transcript for focused analysis.
        """
        # Define patterns that might indicate the call conclusion section
        conclusion_patterns = [
            r'(goodbye|bye|thank you for calling|have a good day|chat with the vet|follow.up|follow up)',
            r'(anything else|any other questions|will that be all|is there anything else)',
            r'(call back|contact us again|reach out if|don\'t hesitate)',
            r'(final|lastly|in conclusion|to summarize|to wrap up)',
            r'(medicine usage|instructions|directions|how to use|how to take)'
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(conclusion_patterns)
        
        # Split transcript into lines
        lines = transcript.split('\n')
        
        # Try to find the likely conclusion section
        # Strategy: Look at the last third of the transcript first
        # If no matches, expand to half the transcript
        start_idx = max(0, int(len(lines) * 0.67))  # Start from last third
        conclusion_section = lines[start_idx:]
        
        # Check if we found any conclusion pattern
        conclusion_found = any(re.search(combined_pattern, line, re.IGNORECASE) for line in conclusion_section)
        
        # If not found in last third, try last half
        if not conclusion_found:
            start_idx = max(0, int(len(lines) * 0.5))  # Start from last half
            conclusion_section = lines[start_idx:]
            
        # If still not found, take last 20 lines or 25% of transcript, whichever is larger
        if not conclusion_found:
            start_idx = max(0, min(len(lines) - 20, int(len(lines) * 0.75)))
            conclusion_section = lines[start_idx:]
        
        # Join lines back into a string
        conclusion_text = '\n'.join(conclusion_section)
        
        # If resulting text is too short, take more context
        if len(conclusion_text.split()) < 50:
            # Take more context, at least 25% of transcript
            start_idx = max(0, int(len(lines) * 0.75))
            conclusion_section = lines[start_idx:]
            conclusion_text = '\n'.join(conclusion_section)
        
        return conclusion_text

    def score_call_conclusion(self, transcript: str) -> Dict[str, Any]:
        """Score the call conclusion section of a transcript."""
        
        # Extract relevant section for focused analysis
        conclusion_section = self.extract_call_conclusion_section(transcript)
        
        # Generate scoring prompt
        scoring_prompt = self.get_call_conclusion_scoring_prompt(conclusion_section)

        # Get scores from LLM
        raw_scores = self._call_llm(scoring_prompt)
        try:
            parsed_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
            # Provide a default response structure if parsing fails
            parsed_scores = {
                "call_conclusion_scores": {},
                "call_conclusion_summary": "Failed to analyze call conclusion due to processing error.",
                "improvement_summary": "Unable to generate improvements due to processing error."
            }
            # Add default scores for each parameter
            for param in self.call_conclusion_params:
                param_key = param['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '')
                parsed_scores["call_conclusion_scores"][param_key] = {
                    "score": 0, 
                    "max_score": param['max_score'],
                    "explanation": "Error evaluating transcript",
                    "strengths": [],
                    "shortcomings": "Unable to analyze this parameter due to processing error.",
                    "improvements": ["Unable to analyze parameter"]
                }

        # Calculate total scores
        parameter_scores = {}
        total_weighted_score = 0
        total_max_score = 0
        
        for param in self.call_conclusion_params:
            param_name = param['name']
            param_key = param_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '')
            max_score = param['max_score']
            
            # Get the raw score
            param_data = parsed_scores["call_conclusion_scores"].get(param_key, {})
            raw_score = param_data.get("score", 0)
            
            # Add up scores
            total_weighted_score += raw_score
            total_max_score += max_score
            
            # Handle shortcomings - ensure it's a string
            shortcomings = param_data.get("shortcomings", "No specific shortcomings identified.")
            # If it's still in array format, convert to single string
            if isinstance(shortcomings, list):
                shortcomings = " ".join(shortcomings)
                # Ensure it's not too long
                if len(shortcomings.split()) > 30:
                    shortcomings = " ".join(shortcomings.split()[:30]) + "..."
            
            # Process improvements to ensure they're in the expected format
            improvements = param_data.get("improvements", [])
            
            # Handle the case where improvements might be in the old format (object with issue/recommendation/example)
            if isinstance(improvements, dict):
                # Convert old format to a simple array with one entry
                if improvements.get("recommendation"):
                    improvements = [improvements.get("recommendation")]
                else:
                    # If no recommendation found, use a generic message
                    improvements = ["No specific improvement provided"]
            
            # Store the parameter score with strengths, shortcomings, and improvements
            parameter_scores[param_key] = {
                "raw_score": raw_score,
                "max_score": max_score,
                "explanation": param_data.get("explanation", ""),
                "strengths": param_data.get("strengths", []),
                "shortcomings": shortcomings,  # Now a string instead of array
                "improvements": improvements,
                "percentage": (raw_score / max_score) * 100 if max_score > 0 else 0
            }
        
        # Compile results
        results = {
            "category": "call_conclusion",
            "parameter_scores": parameter_scores,
            "total_score": total_weighted_score,
            "max_possible_score": total_max_score,
            "percentage_score": (total_weighted_score / total_max_score) * 100 if total_max_score > 0 else 0,
            "summary": parsed_scores.get("call_conclusion_summary", ""),
            "improvement_summary": parsed_scores.get("improvement_summary", "")
        }
        
        return results

    def score_transcript_category5(self, transcript: str) -> Dict[str, Any]:
        """Score Category 5 (Call Conclusion) for the full transcript."""
        # Score the call conclusion
        conclusion_scores = self.score_call_conclusion(transcript)
        
        return conclusion_scores


# Example usage
if __name__ == "__main__":
    # Sample test transcript
    sample_transcript = """
    Vet: So based on what you've told me about Buddy's digestive issues, I recommend switching to our hypoallergenic formula.
    Customer: That sounds good. How much should I give him?
    Vet: For a 15kg dog like Buddy, I'd recommend 1.5 cups twice daily. Mix it gradually with his current food over 7 days.
    Customer: Got it. And how long until we should see improvement?
    Vet: You should start seeing improvement in about 5-7 days. If not, please don't hesitate to reach out.
    Customer: Is there anything else I should know?
    Vet: Just make sure he has plenty of fresh water available. Also, as a reminder, you can always chat with one of our vets through the Supertails app if you have any concerns before your next appointment.
    Customer: That's good to know. Thank you for your help!
    Vet: You're welcome! Is there anything else I can help you with regarding Buddy's previous prescriptions or medications?
    Customer: No, I think we're good for now.
    Vet: Perfect. Remember to follow the dosage instructions I've given, and don't hesitate to reach out if you have any questions about how to use the medicine. Have a great day!
    Customer: You too. Goodbye!
    """
    
    # Initialize scorer with API key (would be loaded from secure config)
    scorer = CallConclusionScorer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Score the transcript
    results = scorer.score_transcript_category5(sample_transcript)
    
    # Print formatted results
    print(json.dumps(results, indent=2))