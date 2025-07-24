#!/usr/bin/env python3

import json
import os
import argparse
import time
import re
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import category scorers - assuming these are in separate files
from category1 import CallScorer
from category2 import PetInformationScorer
from category3 import CommunicationQualityScorer
from category4 import UltraOptimizedTechnicalAssessmentScorer
from category5 import CallConclusionScorer

# Import Excel report generation function
# from excel_report import create_excel_report

# Import knowledge base from separate module to avoid circular imports
from knowledge_base import CloudVeterinaryKnowledgeBase
import config


def parse_call_duration(duration_str):
    """
    Parse call duration string and return total seconds.
    Expected format: "MM:SS" or "H:MM:SS"
    Returns: total seconds as integer, or 0 if parsing fails
    """
    try:
        if not duration_str or duration_str.strip() == "":
            return 0

        # Split by colon
        parts = duration_str.strip().split(":")

        if len(parts) == 2:  # MM:SS format
            minutes, seconds = parts
            total_seconds = int(minutes) * 60 + int(seconds)
        elif len(parts) == 3:  # H:MM:SS format
            hours, minutes, seconds = parts
            total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        else:
            return 0

        return total_seconds
    except (ValueError, AttributeError):
        return 0


def is_call_duration_valid(duration_str, min_seconds=None):
    """
    Check if call duration meets minimum requirement.
    Returns: True if duration >= min_seconds, False otherwise
    """
    if min_seconds is None:
        min_seconds = getattr(config, "MIN_CALL_DURATION_SECONDS", 15)

    duration_seconds = parse_call_duration(duration_str)
    return duration_seconds >= min_seconds


def extract_call_duration_from_transcript(transcript):

    if not transcript:
        return None

    # Extract file name and duration
    file_info_match = re.search(
        r"File:\s*([^\s]+)\.mp3\s+\(Duration:\s+(\d+:\d+)\)", transcript
    )

    if file_info_match:
        return file_info_match.group(2)
    else:
        return None


class UnifiedTranscriptScorer:
    def __init__(
        self,
        api_key: str,
        openai_api_key: str = None,
        model: str = "claude-3-5-sonnet-latest",
    ):
        """Initialize with API key and model type."""
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        self.model = model

        # Initialize all category scorers
        self.category1_scorer = CallScorer(api_key=api_key, model=model)
        self.category2_scorer = PetInformationScorer(api_key=api_key, model=model)
        self.category3_scorer = CommunicationQualityScorer(api_key=api_key, model=model)

        # Initialize cloud-based knowledge base for technical assessment
        try:
            self.knowledge_base = CloudVeterinaryKnowledgeBase(
                collection_name=getattr(
                    config, "COLLECTION_NAME", "veterinary_knowledge"
                ),
                qdrant_url=os.environ.get("QDRANT_URL"),
                qdrant_api_key=os.environ.get("QDRANT_API_KEY"),
            )
        except Exception as e:
            print(f"Warning: Could not initialize cloud knowledge base: {e}")
            print(
                "Make sure QDRANT_URL and QDRANT_API_KEY environment variables are set"
            )
            self.knowledge_base = None

        self.category4_scorer = UltraOptimizedTechnicalAssessmentScorer(
            knowledge_base=self.knowledge_base, openai_api_key=self.openai_api_key
        )

        self.category5_scorer = CallConclusionScorer(api_key=api_key, model=model)

    def load_transcript(self, file_path: str) -> str:
        """Load transcript from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            print(f"Error loading transcript file: {e}")
            return ""

    def validate_call_duration(self, transcript: str) -> tuple[bool, str]:
        """
        Validate if the call meets minimum duration requirements.
        Returns: (is_valid, message)
        """
        # Extract call duration from transcript
        duration_str = extract_call_duration_from_transcript(transcript)

        if not duration_str:
            # If no duration found, we might want to proceed or warn
            warning_msg = "Warning: No call duration found in transcript. Proceeding with scoring."
            print(f"⚠️ {warning_msg}")

            # Check config to see if we should proceed without duration
            proceed_without_duration = getattr(config, "PROCEED_WITHOUT_DURATION", True)
            if proceed_without_duration:
                return True, warning_msg
            else:
                return (
                    False,
                    "Call duration not found in transcript and PROCEED_WITHOUT_DURATION is False",
                )

        # Validate duration
        min_seconds = getattr(config, "MIN_CALL_DURATION_SECONDS", 15)
        is_valid = is_call_duration_valid(duration_str, min_seconds)

        if is_valid:
            actual_seconds = parse_call_duration(duration_str)
            return (
                True,
                f"Call duration ({duration_str} = {actual_seconds}s) meets minimum requirement ({min_seconds}s)",
            )
        else:
            actual_seconds = parse_call_duration(duration_str)
            return (
                False,
                f"Call duration ({duration_str} = {actual_seconds}s) is below minimum requirement ({min_seconds}s)",
            )

    def normalize_technical_score(self, tech_result: dict) -> dict:
        """Normalize technical score result from UltraOptimizedTechnicalAssessmentScorer to match the format of other categories."""
        # Create a normalized parameter_scores dictionary from the new format
        parameter_scores = {}

        for k, v in tech_result.get("parameters", {}).items():
            param_key = k.lower().replace(" ", "_").replace("(", "").replace(")", "")

            # Extract strengths and shortcomings
            strengths = v.get("strengths", [])
            shortcomings = v.get("shortcomings", [])
            improvements = v.get("improvements", [])

            # Format explanation text with strengths and shortcomings
            explanation = ""
            if "entities" in v and v["entities"]:
                explanation += f"Entities found: {', '.join(v['entities'])}\n\n"

            explanation += (
                "Strengths:\n- " + "\n- ".join(strengths)
                if strengths
                else "Strengths: None identified"
            )
            explanation += (
                "\n\nShortcomings:\n- " + "\n- ".join(shortcomings)
                if shortcomings
                else "\n\nShortcomings: None identified"
            )

            # Add comparative summary if available (for medicine parameters)
            comparative_summary = v.get("comparative_summary", "")
            if comparative_summary:
                explanation += f"\n\nComparative Summary:\n{comparative_summary}"

            # Use improvements for non-medicine parameters, comparative summary for medicine parameters
            medicine_params = [
                "Medicine Name",
                "Medicine Prescribed",
                "Medicine Dosage",
            ]
            if k in medicine_params:
                improvements_text = (
                    comparative_summary
                    if comparative_summary
                    else "No comparative summary available"
                )
            else:
                improvements_text = (
                    "\n".join(improvements)
                    if improvements
                    else "No specific improvements identified"
                )

            parameter_scores[param_key] = {
                "raw_score": round(v["score"], 2),
                "max_score": v["max_score"],
                "explanation": explanation,
                "strengths": strengths,
                "shortcomings": shortcomings,
                "percentage": v.get("percentage", 0),
                "improvements": improvements_text,
            }

        # Updated to use the correct fields from the new category4 structure
        normalized = {
            "category": "technical_assessment",
            "parameter_scores": parameter_scores,
            "total_score": tech_result.get(
                "total_score", 0
            ),  # Changed from overall_score to total_score
            "max_possible_score": tech_result.get(
                "max_possible_score", 0
            ),  # Use actual max_possible_score
            "percentage_score": tech_result.get(
                "overall_score", 0
            ),  # overall_score is already a percentage
            "summary": tech_result.get("summary", ""),
            "medicine_findings": tech_result.get("medicine_findings", ""),
        }

        return normalized

    def score_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Score a transcript across all five categories.
        First validates call duration before proceeding.

        Returns a dictionary with scores for each category and overall metrics.
        """
        # Validate call duration first
        is_valid, duration_message = self.validate_call_duration(transcript)

        if not is_valid:
            print(f"❌ {duration_message}")
            return {
                "error": "Call duration validation failed",
                "message": duration_message,
                "transcript_length": len(transcript.split()),
                "validation_failed": True,
            }
        else:
            print(f"✅ {duration_message}")

        # Initialize results structure
        results = {
            "transcript_length": len(transcript.split()),
            "duration_validation": {"is_valid": is_valid, "message": duration_message},
            "categories": {},
            "overall": {
                "total_score": 0,
                "max_possible_score": 0,
                "percentage_score": 0,
            },
        }

        def update_overall(category_key, result):
            """Helper to update overall scores when a category is processed."""
            if not isinstance(result, dict) or "error" in result:
                return

            results["categories"][category_key] = result

            # Add to overall scores if they exist
            if "total_score" in result and "max_possible_score" in result:
                results["overall"]["total_score"] += result["total_score"]
                results["overall"]["max_possible_score"] += result["max_possible_score"]

        # Category 1: Call Introduction
        try:
            # print("Scoring Category 1: Call Introduction...")
            print("\n🎯 Scoring Category 1: Call Introduction...")
            cat1 = self.category1_scorer.score_transcript_category1(transcript)
            update_overall("call_introduction", cat1)
            print(f"✅ Category 1 completed: {cat1.get('percentage_score', 0):.1f}%")
        except Exception as e:
            print(f"Error scoring category 1: {e}")
            results["categories"]["call_introduction"] = {"error": str(e)}

        # Category 2: Pet Information
        try:
            print("\n🎯 Scoring Category 2: Pet Information...")
            cat2 = self.category2_scorer.score_transcript_category2(transcript)
            update_overall("pet_information", cat2)
            print(f"✅ Category 2 completed: {cat2.get('percentage_score', 0):.1f}%")
        except Exception as e:
            print(f"Error scoring category 2: {e}")
            results["categories"]["pet_information"] = {"error": str(e)}

        # Category 3: Communication Quality
        try:
            print("\n🎯 Scoring Category 3: Communication Quality...")
            cat3 = self.category3_scorer.score_transcript_category3(transcript)
            update_overall("communication_quality", cat3)
            print(f"✅ Category 3 completed: {cat3.get('percentage_score', 0):.1f}%")
        except Exception as e:
            print(f"Error scoring category 3: {e}")
            results["categories"]["communication_quality"] = {"error": str(e)}

        # Category 4: Technical Assessment - uses UltraOptimizedTechnicalAssessmentScorer with different output format
        try:
            print("\n🎯 Scoring Category 4: Technical Assessment...")
            raw_tech = self.category4_scorer.score_technical_assessment(transcript)
            cat4 = self.normalize_technical_score(raw_tech)
            update_overall("technical_assessment", cat4)
            print(f"✅ Category 4 completed: {cat4.get('percentage_score', 0):.1f}%")
        except Exception as e:
            print(f"Error scoring category 4: {e}")
            results["categories"]["technical_assessment"] = {"error": str(e)}

        # Category 5: Call Conclusion
        try:
            print("\n🎯 Scoring Category 5: Call Conclusion...")
            cat5 = self.category5_scorer.score_transcript_category5(transcript)
            update_overall("call_conclusion", cat5)
            print(f"✅ Category 5 completed: {cat5.get('percentage_score', 0):.1f}%")
        except Exception as e:
            print(f"Error scoring category 5: {e}")
            results["categories"]["call_conclusion"] = {"error": str(e)}

        # Calculate final percentage
        if results["overall"]["max_possible_score"] > 0:
            results["overall"]["percentage_score"] = round(
                results["overall"]["total_score"]
                / results["overall"]["max_possible_score"]
                * 100,
                2,
            )

        # Generate overall summary
        results["overall"]["summary"] = self._generate_overall_summary(results)

        # Add medicine findings as a separate field if available
        if (
            "technical_assessment" in results["categories"]
            and "medicine_findings" in results["categories"]["technical_assessment"]
        ):
            results["medicine_findings"] = results["categories"][
                "technical_assessment"
            ]["medicine_findings"]

        return results

    def _generate_overall_summary(self, results: Dict[str, Any]) -> str:
        """Generate an overall summary based on scoring results."""
        percentage = results["overall"]["percentage_score"]

        # Determine quality level
        if percentage >= 90:
            quality = "excellent"
        elif percentage >= 80:
            quality = "very good"
        elif percentage >= 70:
            quality = "good"
        elif percentage >= 60:
            quality = "satisfactory"
        else:
            quality = "needs improvement"

        # Identify strengths and areas for improvement
        strengths = []
        improvements = []

        for category_name, data in results["categories"].items():
            if isinstance(data, dict) and "percentage_score" in data:
                category_display = category_name.replace("_", " ").title()
                if data["percentage_score"] >= 80:
                    strengths.append(category_display)
                elif data["percentage_score"] < 60:
                    improvements.append(category_display)

        # Construct summary
        summary = (
            f"Overall call quality is {quality} with a score of {percentage:.1f}%."
        )
        if strengths:
            summary += f" Strengths: {', '.join(strengths)}."
        if improvements:
            summary += f" Areas for improvement: {', '.join(improvements)}."

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save results to a JSON file."""
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
            print(f"✅ Results saved to {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function to run the scoring from command line."""
    script_start_time = time.time()
    start_timestamp = datetime.now()
    print(f"🚀 Script started at: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    parser = argparse.ArgumentParser(description="Score a veterinary call transcript")
    parser.add_argument("transcript_file", help="Path to transcript file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results",
        default="call_scores.json",
    )
    # parser.add_argument('--excel', '-e', help='Output Excel file for report', default='scoring_report.xlsx')
    parser.add_argument(
        "--model", help="LLM model to use", default="claude-3-5-sonnet-latest"
    )
    args = parser.parse_args()

    # Get API keys from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        return

    if not openai_api_key:
        print(
            "⚠️ Warning: OPENAI_API_KEY environment variable not set. Category 4 scoring may be limited."
        )

    # Initialize scorer
    scorer = UnifiedTranscriptScorer(
        api_key=api_key, openai_api_key=openai_api_key, model=args.model
    )

    # Load transcript
    transcript = scorer.load_transcript(args.transcript_file)
    if not transcript:
        print("❌ Error: Failed to load transcript")
        return

    # Score transcript
    print(f"Scoring transcript: {args.transcript_file}")
    results = scorer.score_transcript(transcript)

    # Check if validation failed
    if results.get("validation_failed"):
        print("❌ Transcript processing stopped due to validation failure")
        return

    script_end_time = time.time()
    end_timestamp = datetime.now()
    total_script_time = script_end_time - script_start_time

    print(f"\n🏁 Script completed at: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"⏱️  Total execution time: {total_script_time:.2f} seconds ({total_script_time/60:.2f} minutes)"
    )
    # Save JSON results
    scorer.save_results(results, args.output)

    # Generate Excel report
    # print(f"Generating Excel report: {args.excel}")
    # try:
    #     create_excel_report(results, args.excel, args.transcript_file)
    #     print(f"✅ Excel report saved to {args.excel}")
    # except Exception as e:
    #     print(f"❌ Error generating Excel report: {e}")

    # Print summary
    print("\n===== SCORING SUMMARY =====")
    print(results["overall"]["summary"])
    print(
        f"Total Score: {results['overall']['total_score']} / {results['overall']['max_possible_score']}"
    )
    print(f"Percentage: {results['overall']['percentage_score']}%")
    print("\nCategory Scores:")
    for cat, data in results["categories"].items():
        if isinstance(data, dict) and "percentage_score" in data:
            print(f"  {cat.replace('_', ' ').title()}: {data['percentage_score']}%")

    # Print medicine findings if available
    if "medicine_findings" in results:
        print("\n===== MEDICINE FINDINGS =====")
        print(results["medicine_findings"])


if __name__ == "__main__":
    main()
