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

class CommunicationQualityScorer:
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
        
        # Create a mapping for communication quality parameters
        self.communication_params = [
            param for param in self.parameters_config 
            if param['name'] in [
                "Tone & Voice Modulation", 
                "Clear Communication",
                "Engagement & Rapport Building",
                "Patience & Attentiveness",
                "Empathy & Compassion"
            ]
        ]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with structured prompt and handle retries."""
        try:
            system_prompt = """You are an expert in evaluating communication quality in veterinary support calls.
            Analyze transcripts to assess tone, clarity, engagement, patience, and empathy.
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

    def get_communication_scoring_prompt(self, transcript: str) -> str:
        """Generate scoring prompt for communication quality parameters."""
        prompt_parts = [
            f"""
            Carefully analyze this veterinary support call transcript to evaluate the vet's communication quality:

            TRANSCRIPT:
            {transcript}

            Score the following communication quality parameters based on their specific criteria:
            """
        ]
        
        # Add scoring criteria for each parameter
        for param in self.communication_params:
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
            prompt_parts.append(f"- 0: Poor execution")
            mid_point = max_score / 2
            prompt_parts.append(f"- {mid_point:.1f}: Adequate execution")
            prompt_parts.append(f"- {max_score}: Excellent execution")
            prompt_parts.append("")
        
        # Add specific guidance for each parameter
        prompt_parts.append("""
        Specific criteria for each parameter:

        1. Tone & Voice Modulation:
           - Maintain a warm, friendly tone throughout the call
           - Use appropriate voice modulation to engage the customer
           - Avoid monotonous speaking or sounding disinterested
           - Adjust tone based on the customer's emotional state

        2. Clear Communication:
           - Use simple, non-technical language when explaining concepts
           - Avoid medical jargon unless necessary, and explain it when used
           - Speak at an appropriate pace and enunciate clearly
           - Confirm understanding when discussing complex information

        3. Engagement & Rapport Building:
           - Use the pet's name naturally in conversation
           - Show interest in the customer's concerns
           - Ask appropriate follow-up questions
           - Maintain conversation flow

        4. Patience & Attentiveness:
           - Allow customer to fully express concerns without interruption
           - Remain calm and professional when customers are upset or confused
           - Repeat or rephrase information when needed
           - Address all concerns raised by the customer

        5. Empathy & Compassion:
           - Express genuine concern for the pet's wellbeing
           - Acknowledge the customer's emotions and concerns
           - Use phrases that show understanding of the situation
           - Demonstrate care in tone and word choice
        """)
        
        # Add JSON response format instructions with strengths, shortcomings and improvements
        prompt_parts.append("""
        For each parameter, provide:
        1. Clear strengths with specific evidence from the transcript (max 2-3 bullets)
        2. For any parameter not receiving a perfect score, identify the specific shortcoming with:
           - A direct quote or specific moment from the transcript showing the gap
           - What exactly should have been said or done instead
           - FORMAT: "At [specific moment/quote], should have [specific action/words], [provide some suggestive example]" (maximum 30 words)
        3. Provide a crisp, professional improvement suggestion summarizing the issue and corrective action.
        
        Keep all points extremely concise.

        Format your response as a structured JSON with the following format:
        {
            "communication_quality_scores": {
        """)
        
        # Add expected structure for each parameter - MODIFIED to add improvements as a list of strings
        for param in self.communication_params:
            param_key = param['name'].lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
            prompt_parts.append(f"""        "{param_key}": {{
                    "score": X,
                    "max_score": {param['max_score']},
                    "explanation": "Brief explanation with evidence from transcript",
                    "strengths": ["Specific strength 1", "Specific strength 2"],
                    "shortcomings": "A descriptive paragraph of shortcomings in maximum 30 words",
                    "improvements": ["Actionable, concise improvement recommendation"]
                }},""")
        
        # Close the JSON structure
        prompt_parts.append("""
            },
            "communication_quality_summary": "Brief overall assessment of communication quality",
            "improvement_summary": "3-5 bullet-point priority improvements, each with one brief example"
        }
        
        Ensure you provide specific, actionable improvements for each parameter that has shortcomings.
        Keep all points extremely concise.
        Use bullet-style formatting for strengths list.
        Ensure shortcomings are in a descriptive paragraph format with maximum 30 words and stick to the format that is provided.
        Ensure your response is a valid JSON object.
        """)
        
        return "".join(prompt_parts)

    def score_communication_quality(self, transcript: str) -> Dict[str, Any]:
        """Score the communication quality of a transcript."""
        
        # Generate scoring prompt
        scoring_prompt = self.get_communication_scoring_prompt(transcript)

        # Get scores from LLM
        raw_scores = self._call_llm(scoring_prompt)
        try:
            parsed_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON")
            # Provide a default response structure if parsing fails
            parsed_scores = {
                "communication_quality_scores": {},
                "communication_quality_summary": "Failed to analyze communication quality due to processing error.",
                "improvement_summary": "Unable to generate improvements due to processing error."
            }
            for param in self.communication_params:
                param_key = param['name'].lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
                parsed_scores["communication_quality_scores"][param_key] = {
                    "score": 0,
                    "max_score": param['max_score'],
                    "explanation": "Error evaluating transcript",
                    "strengths": [],
                    "shortcomings": "Unable to analyze this parameter due to processing error.",
                    "improvements": ["Retry analysis"]
                }

        # Process and score
        parameter_scores = {}
        total_weighted_score = 0
        total_max_score = 0

        for param in self.communication_params:
            param_name = param['name']
            param_key = param_name.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
            max_score = param['max_score']

            param_data = parsed_scores["communication_quality_scores"].get(param_key, {})
            raw_score = param_data.get("score", 0)

            total_weighted_score += raw_score
            total_max_score += max_score

            # Improvements: accept as list of concise strings
            improvements = param_data.get("improvements", [])
            # Handle different formats that might be returned
            if isinstance(improvements, str):
                improvements = [improvements]
            
            # Shortcomings: now a descriptive paragraph string limited to 30 words
            shortcomings = param_data.get("shortcomings", "No shortcomings identified.")
            # If somehow we still get a list, convert it to a descriptive string
            if isinstance(shortcomings, list):
                shortcomings = " ".join(shortcomings)
                # Limit to 30 words
                words = shortcomings.split()
                if len(words) > 30:
                    shortcomings = " ".join(words[:30]) + "..."
            
            parameter_scores[param_key] = {
                "raw_score": raw_score,
                "max_score": max_score,
                "explanation": param_data.get("explanation", ""),
                "strengths": param_data.get("strengths", []),
                "shortcomings": shortcomings,  # Now a string, not a list
                "improvements": improvements,
                "percentage": (raw_score / max_score) * 100 if max_score > 0 else 0
            }

        # Compile final result
        results = {
            "category": "communication_quality",
            "parameter_scores": parameter_scores,
            "total_score": total_weighted_score,
            "max_possible_score": total_max_score,
            "percentage_score": (total_weighted_score / total_max_score) * 100 if total_max_score > 0 else 0,
            "summary": parsed_scores.get("communication_quality_summary", ""),
            "improvement_summary": parsed_scores.get("improvement_summary", "")
        }

        return results

    
    def extract_transcript_segments(self, transcript: str, segment_length: int = 50) -> List[str]:
        """
        Break the transcript into manageable segments for analysis.
        This helps when dealing with long transcripts that might exceed context windows.
        
        Args:
            transcript: The full call transcript
            segment_length: Approximate number of lines per segment
            
        Returns:
            List of transcript segments
        """
        lines = transcript.strip().split('\n')
        segments = []
        
        # Skip empty lines
        lines = [line for line in lines if line.strip()]
        
        # Break into segments
        for i in range(0, len(lines), segment_length):
            segment = '\n'.join(lines[i:i + segment_length])
            segments.append(segment)
            
        return segments
    
    def score_transcript_category3(self, transcript: str) -> Dict[str, Any]:
        """Score Category 3 (Communication Quality) for the full transcript."""
        # For communication quality, we need to analyze the entire transcript
        # But we may need to break it into segments if it's very long
        
        # Check if transcript is too long
        if transcript.count('\n') > 100:
            segments = self.extract_transcript_segments(transcript)
            
            # Score each segment
            segment_scores = []
            for segment in segments:
                segment_scores.append(self.score_communication_quality(segment))
            
            # Combine segment scores (average of scores)
            combined_scores = self._combine_segment_scores(segment_scores)
            return combined_scores
        else:
            # Score the whole transcript directly
            return self.score_communication_quality(transcript)
    
    def _combine_segment_scores(self, segment_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine scores from multiple transcript segments.
        Takes the mean score for each parameter across segments.
        Ensures only one improvement is included per parameter.
        """
        if not segment_scores:
            return {
                "category": "communication_quality",
                "parameter_scores": {},
                "total_score": 0,
                "max_possible_score": 0,
                "percentage_score": 0,
                "summary": "No segments scored",
                "improvement_summary": "No segments to improve"
            }
        
        # Initialize combined structure
        combined_result = {
            "category": "communication_quality",
            "parameter_scores": {},
            "total_score": 0,
            "max_possible_score": 0,
            "percentage_score": 0,
            "summary": "",
            "improvement_summary": ""
        }
        
        # Combine parameter scores
        all_params = set()
        for segment in segment_scores:
            if "parameter_scores" in segment:
                all_params.update(segment["parameter_scores"].keys())
        
        # For each parameter, average the scores across segments
        for param in all_params:
            param_segments = [
                segment["parameter_scores"].get(param, {
                    "raw_score": 0, 
                    "max_score": 0, 
                    "explanation": "", 
                    "strengths": [], 
                    "shortcomings": "No analysis available.",
                    "improvements": []
                }) 
                for segment in segment_scores 
                if "parameter_scores" in segment
            ]
            
            # Calculate averages
            raw_scores = [p.get("raw_score", 0) for p in param_segments if p]
            max_scores = [p.get("max_score", 0) for p in param_segments if p]
            explanations = [p.get("explanation", "") for p in param_segments if p and p.get("explanation")]
            
            # Combine strengths while removing duplicates
            all_strengths = []
            
            for p in param_segments:
                if p:
                    strengths = p.get("strengths", [])
                    
                    # Make sure we're handling strings, not lists when adding to all_strengths
                    for strength in strengths:
                        if strength and isinstance(strength, str) and strength not in all_strengths:
                            all_strengths.append(strength)
            
            # For shortcomings, combine into a single descriptive paragraph within 30 words
            all_shortcomings = []
            for p in param_segments:
                if p:
                    shortcoming = p.get("shortcomings", "")
                    # If it's a string, add it to our collection
                    if shortcoming and isinstance(shortcoming, str):
                        all_shortcomings.append(shortcoming)
                    # If somehow it's a list, convert to strings and add
                    elif isinstance(shortcoming, list):
                        for sc in shortcoming:
                            if sc and isinstance(sc, str):
                                all_shortcomings.append(sc)
            
            # Create a combined shortcomings paragraph limited to 30 words
            combined_shortcomings = ""
            if all_shortcomings:
                # Use the first shortcoming as it's already in descriptive paragraph format
                combined_shortcomings = all_shortcomings[0]
                # Ensure it's within 30 words
                words = combined_shortcomings.split()
                if len(words) > 30:
                    combined_shortcomings = " ".join(words[:30]) + "..."
            else:
                combined_shortcomings = "No significant shortcomings identified across transcript segments."
            
            # Select only one improvement (the best one)
            all_improvements_candidates = []
            for p in param_segments:
                if p:
                    improvements = p.get("improvements", [])
                    
                    if isinstance(improvements, list):
                        for improvement in improvements:
                            if improvement and isinstance(improvement, str) and improvement not in all_improvements_candidates:
                                all_improvements_candidates.append(improvement)
                    elif isinstance(improvements, str) and improvements not in all_improvements_candidates:
                        all_improvements_candidates.append(improvements)
            
            # Select only the first/best improvement
            selected_improvement = []
            if all_improvements_candidates:
                # Select the first improvement that doesn't start with a bullet point for cleaner output
                clean_improvements = [imp for imp in all_improvements_candidates if not imp.strip().startswith('•')]
                if clean_improvements:
                    selected_improvement = [clean_improvements[0]]
                else:
                    # If all have bullet points, just take the first one
                    selected_improvement = [all_improvements_candidates[0]]
            
            # Store averaged results
            combined_result["parameter_scores"][param] = {
                "raw_score": sum(raw_scores) / len(raw_scores) if raw_scores else 0,
                "max_score": max(max_scores) if max_scores else 0,
                "explanation": "Combined from multiple segments: " + "; ".join(explanations[:3]) if explanations else "",
                "strengths": all_strengths[:3],  # Limit to top 3 strengths
                "shortcomings": combined_shortcomings,  # Now a descriptive paragraph
                "improvements": selected_improvement,  # Only include one improvement
                "percentage": (sum(raw_scores) / len(raw_scores) / max(max_scores) * 100) if raw_scores and max_scores and max(max_scores) > 0 else 0
            }
        
        # Calculate combined totals
        combined_result["total_score"] = sum(param["raw_score"] for param in combined_result["parameter_scores"].values())
        combined_result["max_possible_score"] = sum(param["max_score"] for param in combined_result["parameter_scores"].values())
        
        # Calculate percentage
        if combined_result["max_possible_score"] > 0:
            combined_result["percentage_score"] = (combined_result["total_score"] / combined_result["max_possible_score"]) * 100
        else:
            combined_result["percentage_score"] = 0
            
        # Combine summaries - make sure we're working with strings
        summaries = [segment.get("summary", "") for segment in segment_scores if segment.get("summary") and isinstance(segment.get("summary"), str)]
        combined_result["summary"] = "Overall: " + " ".join(summaries[:3]) if summaries else "No summary available"
        
        # Combine improvement summaries - make sure we're working with strings
        improvement_summaries = [segment.get("improvement_summary", "") for segment in segment_scores 
                            if segment.get("improvement_summary") and isinstance(segment.get("improvement_summary"), str)]
        combined_result["improvement_summary"] = "Priority improvements: " + " ".join(improvement_summaries[:3]) if improvement_summaries else "No improvement summary available"
        
        return combined_result


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
    scorer = CommunicationQualityScorer(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Score the transcript
    results = scorer.score_transcript_category3(sample_transcript)
    
    # Print formatted results
    print(json.dumps(results, indent=2))