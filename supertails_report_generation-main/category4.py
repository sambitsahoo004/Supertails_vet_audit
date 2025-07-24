import json
import openai
from typing import List, Dict, Any, Tuple, Optional
import re
import os
from dotenv import load_dotenv
from config import PARAMETERS_CONFIG
from knowledge_base import CloudVeterinaryKnowledgeBase
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


class UltraOptimizedTechnicalAssessmentScorer:
    """
    Ultra-optimized class for scoring Category 4: Technical Assessment parameters
    using only 2-3 LLM calls total while preserving all original logic.

    Reduction from 9-12 calls to 2-3 calls using mega-prompts and batch processing.
    """

    def __init__(
        self,
        knowledge_base: Optional[CloudVeterinaryKnowledgeBase] = None,
        openai_api_key: Optional[str] = None,
        config: Dict = None,
        llm_model: str = "gpt-4o",
    ):
        """
        Initialize the technical assessment scorer with ultra-optimized scoring capabilities.
        """
        self.kb = knowledge_base
        self.config = config if config else PARAMETERS_CONFIG
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model

        # Extract technical assessment parameters from config
        self.technical_parameters = self._extract_technical_parameters()

        # Define which parameters use knowledge base comparison
        self.kb_comparison_params = [
            "Medicine Name",
            "Medicine Prescribed",
            "Medicine Dosage",
        ]

        # Define pharmacy-related parameters for lenient scoring
        self.pharmacy_params = [
            "Medicine Name",
            "Medicine Prescribed",
            "Medicine Dosage",
        ]

        # Parameter descriptions for better LLM context
        self.parameter_descriptions = {
            "Diet Confirmation": "Verification of the pet's current diet including type, brand, and feeding routine.",
            "Food Brand Name": "Specific brand names of pet food mentioned during the consultation.",
            "Technical Parameters": "Medical and clinical parameters assessed during examination like weight, temperature, etc.",
            "Treatment Plan and Instructions PSP": "Detailed treatment protocols, steps, and post-treatment care instructions.",
            "Medicine Name": "Names of specific medications or drugs recommended or discussed.",
            "Medicine Prescribed": "Information about prescriptions issued including proper prescription protocols.",
            "Medicine Dosage": "Dosing information including quantity, frequency, and administration method.",
        }

    def _extract_technical_parameters(self) -> Dict[str, float]:
        """Extract Category 4 parameters from config."""
        all_parameters = self.config.get("parameters", [])
        technical_params = {}

        target_params = [
            "Diet Confirmation",
            "Food Brand Name",
            "Technical Parameters",
            "Treatment Plan and Instructions PSP",
            "Medicine Name",
            "Medicine Prescribed",
            "Medicine Dosage",
        ]

        for param in all_parameters:
            param_name = param.get("name")
            if param_name in target_params or param_name in [
                p.lower() for p in target_params
            ]:
                max_score = param.get("max_score", 3)
                weight = max_score / 100
                standardized_name = param_name.title().replace(" And ", " and ")
                technical_params[standardized_name] = weight

        if not technical_params:
            technical_params = {
                "Diet Confirmation": 0.03,
                "Food Brand Name": 0.03,
                "Technical Parameters": 0.04,
                "Treatment Plan and Instructions PSP": 0.03,
                "Medicine Name": 0.04,
                "Medicine Prescribed": 0.04,
                "Medicine Dosage": 0.04,
            }

        return technical_params

    def _get_parameter_config(self, parameter_name: str) -> Dict:
        """Get configuration for a specific parameter."""
        parameter_variants = [
            parameter_name,
            parameter_name.lower(),
            parameter_name.title(),
            parameter_name.replace(" and ", " And "),
            parameter_name.replace(" And ", " and "),
            parameter_name.replace("PSP", "").strip(),
            parameter_name.replace("psp", "").strip(),
            # Handle the specific case
            (
                "Treatment Plan and Instructions PSP"
                if "treatment plan" in parameter_name.lower()
                else parameter_name
            ),
        ]

        # Also check if any config parameter matches our variants
        for param in self.config.get("parameters", []):
            param_name = param.get("name", "")

            # Direct match check
            if param_name in parameter_variants:
                return param

            # Reverse match check (config name matches our parameter)
            config_variants = [
                param_name,
                param_name.lower(),
                param_name.title(),
                param_name.replace(" and ", " And "),
                param_name.replace(" And ", " and "),
            ]

            if parameter_name in config_variants or parameter_name.lower() in [
                v.lower() for v in config_variants
            ]:
                return param

        # FIXED: Use correct default max_score based on parameter type
        # Map parameter names to their correct max_scores from your config
        default_max_scores = {
            "Diet Confirmation": 3,
            "Food Brand Name": 3,
            "Technical Parameters": 4,
            "Treatment Plan and Instructions PSP": 3,  # <-- CORRECT VALUE
            "Medicine Name": 4,
            "Medicine Prescribed": 4,
            "Medicine Dosage": 4,
        }

        # Find the correct max_score for this parameter
        correct_max_score = 3  # safe default
        for param_key, max_score in default_max_scores.items():
            if (
                param_key.lower() in parameter_name.lower()
                or parameter_name.lower() in param_key.lower()
            ):
                correct_max_score = max_score
                break

        return {
            "name": parameter_name,
            "max_score": correct_max_score,  # <-- NOW USES CORRECT VALUE
            "subparameters": [
                {"name": f"{parameter_name} assessment", "weight": correct_max_score}
            ],
        }

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, prompt: str, response_format: str = "json_object") -> Dict:
        """Call LLM with structured prompt and handle retries."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a veterinary quality assessment expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=(
                    {"type": response_format}
                    if response_format == "json_object"
                    else None
                ),
                temperature=0.3,
            )

            content = response.choices[0].message.content
            if response_format == "json_object":
                return json.loads(content)
            return content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            if response_format == "json_object":
                return {}
            return ""

    def _retrieve_knowledge_batch(self, all_queries: List[str]) -> List[Dict]:
        """Retrieve knowledge for all queries in batch."""
        if not self.kb or not all_queries:
            return []

        try:
            print("=" * 80)
            print("KNOWLEDGE BASE QUERIES BEING SENT:")
            print("=" * 80)
            for i, query in enumerate(all_queries, 1):
                print(f"Query {i}: {query}")
            print("=" * 80)

            combined_results = []
            for query in all_queries:
                results = self.kb.search(query, limit=2)  # Reduced limit for efficiency
                print(f"\nResults for query '{query}':")
                for j, result in enumerate(results):
                    print(
                        f"  Result {j+1}: {result['text'][:100]}..."
                    )  # First 100 chars
                combined_results.extend(results)

            # Remove duplicates and limit total results
            unique_results = []
            seen_texts = set()
            for result in combined_results:
                if (
                    result["text"] not in seen_texts and len(unique_results) < 12
                ):  # Max 12 knowledge chunks
                    seen_texts.add(result["text"])
                    unique_results.append(result)

            print(f"\nFINAL UNIQUE KNOWLEDGE RESULTS ({len(unique_results)} total):")
            print("=" * 80)
            for i, result in enumerate(unique_results, 1):
                print(f"Knowledge Chunk {i}:")
                print(f"  Text: {result['text'][:200]}...")  # First 200 chars
                print(f"  Score: {result.get('score', 'N/A')}")
                print("-" * 40)
            print("=" * 80)

            return unique_results
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return []

    def _mega_comprehensive_analysis(self, transcript: str) -> Dict:
        """
        SINGLE MEGA LLM CALL: Analyze ALL technical parameters simultaneously
        using advanced chain of thoughts and parallel processing logic with enhanced
        contextual query generation built into the mega-prompt.
        """

        # Create parameter-specific criteria for all parameters with enhanced instructions
        parameter_criteria = {
            "Medicine Dosage": """
            MEDICINE DOSAGE ULTRA-STRICT CRITERIA:
            CRITICAL: ONLY score based on medications/products that are EXPLICITLY MENTIONED BY NAME in the transcript.
            DO NOT assume or invent any medications not explicitly mentioned.
            
            Dosage information MUST include ACTUAL references to quantity, frequency, or administration method FOR EXPLICITLY MENTIONED MEDICATIONS ONLY.
            
            Examples that qualify: 
            - "Give 10mg twice daily" (for a named medication)
            - "Apply small amount daily" (for a named product)  
            - "Use after shampooing" (for a named treatment)
            - "Works for three months" (duration for named medication)
            - "1 tablet per 10kg body weight" (for named medication)
            
            Examples that DON'T qualify: 
            - "I'll tell you how to use it" 
            - "Instructions are on the label"
            - Generic statements without specific medication names
            
            SCORING: 0=No actual dosage info for named medications, 1=Extremely vague, 2-3=Partial dosage info for named medications, 4-5=Complete dosage info for named medications
            """,
            "Medicine Name": """
            MEDICINE NAME ENHANCED LENIENT CRITERIA:
            FIRST: Carefully extract ALL medications/products that are EXPLICITLY MENTIONED BY NAME in the transcript.
            THEN: Assess the completeness and specificity of medication naming.
            
            Can include:
            - Exact product names (e.g., "Brevicto", "ECPET", "Praziplus")
            - Brand names with specifications (e.g., "MSD Brevicto for 20 to 40 kgs")
            - Product categories when specific brands aren't mentioned (e.g., "anti-tick and flea shampoo")
            - Descriptive names for medical products (e.g., "ear cleaning solution")
            
            Examples that score high:
            - "Brevicto" (specific product name)
            - "MSD Brevicto for 20 to 40 kgs" (complete product specification)
            - "ECPET deworming tablet" (specific named product with category)
            
            SCORING: 0=Nothing mentioned, 1=Vague reference, 2-3=Product category or partial name, 4-5=Specific complete names
            """,
            "Medicine Prescribed": """
            MEDICINE PRESCRIBED ENHANCED LENIENT CRITERIA:
            Focus on prescription processes for EXPLICITLY MENTIONED medications/products only.
            
            Includes verification steps, prescription creation mentions, or pharmacy coordination FOR NAMED MEDICATIONS.
            
            Examples that qualify:
            - "Sending prescription to pharmacy" (for named medication)
            - "Verification call for pharmacy product" (for specific product)
            - "Creating prescription" (for mentioned medication)
            - "We have to create a prescription from the companies" (for specific product)
            - "I will send a prescription to your WhatsApp" (for discussed medication)
            
            Look for prescription workflow elements:
            1. Verification processes for specific medications
            2. Prescription creation mentions
            3. Documentation requirements for named products
            4. Pharmacy coordination for specific medications
            
            SCORING: 0=No prescription process mentioned, 1=Minimal reference, 2-3=Clear prescription process, 4-5=Complete prescription workflow
            """,
            "Diet Confirmation": """
            DIET CONFIRMATION CRITERIA:
            Verification of pet's current diet, feeding routine, food brands, or dietary discussions.
            Look for specific mentions of:
            - Current food brands or types
            - Feeding schedules or routines
            - Dietary restrictions or preferences
            - Food-related health discussions
            - Administration methods involving food (e.g., "mixed with food")
            
            SCORING: 0=No diet discussion, 1=Minimal mention, 2-3=Some dietary info, 4-5=Comprehensive diet discussion
            """,
            "Food Brand Name": """
            FOOD BRAND NAME CRITERIA:
            Specific pet food brand names mentioned during consultation.
            Look for actual brand names, not just generic food references.
            
            Examples that qualify:
            - "Royal Canin"
            - "Hill's Science Diet"
            - "Pedigree"
            
            Examples that don't qualify:
            - "dry food"
            - "kibble"
            - "pet food"
            
            SCORING: 0=No brands mentioned, 1=Vague food reference, 2-3=Food type mentioned, 4-5=Specific brand names
            """,
            "Technical Parameters": """
            TECHNICAL PARAMETERS CRITERIA:
            Medical/clinical parameters like weight, temperature, age, breed, physical examination details.
            
            Look for specific mentions of:
            - Pet weight (e.g., "24 to 25 kg")
            - Pet age (e.g., "five years old")
            - Breed information (e.g., "Beagle")
            - Gender (e.g., "male")
            - Physical examination findings
            - Vital signs or measurements
            - Clinical observations
            
            SCORING: 0=No clinical data, 1=Minimal info, 2-3=Some clinical details, 4-5=Comprehensive clinical assessment
            """,
            "Treatment Plan and Instructions PSP": """
            TREATMENT PLAN CRITERIA:
            Detailed treatment protocols, post-treatment care, follow-up instructions, administration guidance.
            
            Look for:
            - Step-by-step treatment instructions
            - Administration methods (e.g., "directly in mouth or mixed with food")
            - Duration of treatment (e.g., "works for three months")
            - Follow-up instructions (e.g., "repeat if ticks and fleas come back")
            - Future care guidance
            - Monitoring instructions
            
            SCORING SYSTEM (VERY LENIENT):
            - 5 = Comprehensive: Duration + Frequency + Administration + Follow-up/Monitoring
            - 4 = Very Good: Any 3 elements (e.g., "2 weeks" + "once daily" + "mix with food")
            - 3 = Good: Any 2 strong elements (e.g., "once daily" + "mix with food" OR "2 weeks" + "see improvement")
            - 2 = Fair: At least 1 strong element + 1 basic element (e.g., frequency + basic method)
            - 1 = Minimal: At least 1 clear instruction element (just frequency OR just method OR just duration)
            - 0 = Nothing: Absolutely no treatment guidance of any kind
            """,
        }

        # Format all parameter info for the mega prompt
        params_info = "\n\n".join(
            [
                f"PARAMETER: {param}\nDESCRIPTION: {self.parameter_descriptions.get(param, param)}\n{criteria}"
                for param, criteria in parameter_criteria.items()
            ]
        )

        prompt = f"""
        You are a veterinary quality assessment expert. Perform SIMULTANEOUS COMPREHENSIVE ANALYSIS 
        of ALL technical assessment parameters using advanced chain of thoughts reasoning with 
        ENHANCED CONTEXTUAL QUERY GENERATION built into the analysis process.
        
        TRANSCRIPT TO ANALYZE:
        {transcript}
        
        PARAMETERS TO ASSESS SIMULTANEOUSLY:
        {params_info}
        
        MEGA CHAIN OF THOUGHTS ANALYSIS - Complete ALL steps for ALL parameters:
        
        STEP 1: COMPREHENSIVE PET & HEALTH CONTEXT EXTRACTION
        Extract complete pet and health information in ONE go for all parameters:
        
        A) PET DETAILS:
        - Name, species, breed, age, weight, gender, neutered status
        - Indoor/outdoor lifestyle
        - Physical characteristics mentioned
        
        B) HEALTH CONTEXT:
        - Primary health concerns and symptoms
        - Diagnosed conditions and severity
        - Duration of health issues
        - Affected body parts/systems
        - Previous treatments attempted
        
        C) TREATMENT CONTEXT:
        - Purpose of current consultation (prevention/treatment/maintenance)
        - Target conditions or parasites being addressed
        - Administration preferences or challenges mentioned
        - Treatment duration discussed
        - Follow-up requirements
        
        D) OWNER CONTEXT:
        - Main concerns expressed by owner
        - Experience level with pet medication
        - Cost sensitivity if mentioned
        - Previous medication experiences or issues
        
        E) EXPLICITLY MENTIONED MEDICATIONS:
        CRITICAL: Include ONLY medications/products explicitly mentioned by name in the transcript.
        DO NOT assume, invent, or infer any medications not explicitly stated.
        For each medication mentioned, note:
        - Exact name as stated
        - Purpose/indication mentioned
        - Any dosage information provided
        - Administration method discussed
        
        STEP 2: CONTEXTUAL KNOWLEDGE QUERIES GENERATION
        Using the extracted context, generate 12 HIGHLY SPECIFIC and CONTEXTUAL queries (4 for each medicine parameter):
        
        QUERY GENERATION PRINCIPLES:
        - Incorporate pet species, breed, weight, and age when available
        - Include specific health conditions and symptoms
        - Reference exact medication names mentioned in transcript
        - Consider treatment purpose (prevention vs treatment)
        - Include administration context and owner concerns
        - Make queries as specific as possible to this exact case
        
        For Medicine Name queries (4 queries):
        Template: "[Medication name] for [specific condition] in [pet descriptor with weight/age/breed]"
        Example: "Brevicto flea tick prevention 25kg adult male Beagle outdoor dog"
        Example: "ECPET deworming tablet multiple dogs household different weights"
        
        Focus on:
        1. Specific medication efficacy for this pet's condition and characteristics
        2. Medication suitability for this pet's breed, weight, and age
        3. Comparison with alternatives for this specific case
        4. Safety considerations for this pet's profile
        
        For Medicine Dosage queries (4 queries):
        Template: "[Medication name] dosage [pet weight] [condition] [administration context]"
        Example: "Brevicto 20-40kg tablet dosing frequency duration outdoor dog tick prevention"
        Example: "ECPET deworming dosage calculation 10kg 15kg 25kg multiple dogs"
        
        Focus on:
        1. Weight-specific dosing for this pet
        2. Frequency and duration for this condition
        3. Administration method suitable for this owner/pet
        4. Safety margins and monitoring for this pet
        
        For Medicine Prescribed queries (4 queries):
        Template: "[Medication name] prescription requirements [pet type] [condition] [context]"
        Example: "Brevicto prescription documentation requirements outdoor dog tick prevention"
        Example: "Multiple medication prescription protocol ECPET Brevicto same household"
        
        Focus on:
        1. Prescription requirements for these specific medications
        2. Documentation needs for this treatment plan
        3. Pharmacy coordination for this case
        4. Regulatory compliance for these medications
        
        STEP 3: PARALLEL SEGMENT EXTRACTION WITH CONTEXT
        For each parameter, extract 2-3 most relevant transcript segments considering:
        - Pet-specific context (breed, weight, age, lifestyle)
        - Health condition context (symptoms, severity, duration)
        - Treatment context (prevention vs treatment, administration challenges)
        - Owner context (concerns, experience level)
        
        STEP 4: CONTEXTUAL ENTITY EXTRACTION
        Extract entities for each parameter with full context:
        - For Medicine Dosage: Weight-specific dosage info for explicitly mentioned medications
        - For Medicine Name: Medication names with indication and pet suitability context
        - For Medicine Prescribed: Prescription processes with pet and medication context
        - For other parameters: Relevant entities with pet health context
        
        STEP 5: CONTEXTUAL SCORING AND ASSESSMENT
        Score each parameter (0-5) considering:
        - Completeness of information for this specific pet case
        - Relevance to pet's breed, weight, age, and condition
        - Appropriateness of treatment for this specific situation
        - Quality of guidance provided for this owner's context
        
        STEP 6: CONTEXTUAL STRENGTHS AND SHORTCOMINGS
        For each parameter, identify strengths and shortcomings considering:
        - How well the information fits this specific pet's needs
        - Whether advice is appropriate for this owner's experience level
        - If medication choices suit this pet's characteristics
        - Whether dosing guidance is adequate for this pet's weight/age
        For NON-MEDICINE parameters (Diet Confirmation, Food Brand Name, Technical Parameters, Treatment Plan and Instructions PSP):
        - Generate specific improvement suggestions based on identified shortcomings
        - Focus on actionable recommendations that could enhance the consultation quality
        - Consider what additional information or approach could have been better
        For MEDICINE parameters (Medicine Name, Medicine Prescribed, Medicine Dosage):
        - Only identify shortcomings without improvement suggestions (handled separately)
        CRITICAL SHORTCOMING REQUIREMENTS - MANDATORY COMPLIANCE:
        ❌ FORBIDDEN: "No specific shortcomings identified" or similar generic responses
        ✅ REQUIRED: For ANY score < 5, provide EXACTLY ONE specific shortcoming with transcript example and solution (MAX 30 WORDS)   
        MANDATORY FORMAT: "[Action verb] [specific element].[problem description]. [specific example of how it should have been done]."      
        VARY OPENING VERBS: Lacked, Overlooked, Omitted, Skipped, Avoided, Insufficient, Unclear, Absent, Vague, Incomplete  
        EXAMPLE: "Overlooked weight documentation. lacks precise measurement. Should have recorded exact weight for accurate dosing, like-[Give an example like how it could have been done]."    
        VALIDATION: Before submitting response, verify EVERY parameter with score < 5 has a specific shortcoming with transcript example and solution.
        If you find "No specific shortcomings identified" anywhere, you MUST replace it with a proper shortcoming following the format above.
        
        STEP 7: CONTEXTUAL SUMMARY GENERATION
        Generate summaries that reflect:
        - Overall assessment quality for this specific pet case
        - How well the consultation addressed this pet's unique needs
        - Whether medication choices were appropriate for this situation
        - Quality of guidance provided considering owner's context
        
        Return complete analysis as JSON with this structure:
        {{
            "global_pet_details": {{
                "pet_name": "Name or unknown",
                "pet_type": "Species", 
                "breed": "Breed or unknown",
                "age": "Age with units",
                "weight": "Weight with units",
                "gender": "Male/Female if mentioned",
                "neutered": true/false,
                "indoor_outdoor": "Indoor/Outdoor/Both", 
                "conditions": ["List conditions"],
                "medications": ["ONLY explicitly mentioned medications by exact name"]
            }},
            "explicitly_mentioned_medications": ["List of ALL medications/products mentioned by name in transcript"],
            "global_knowledge_queries": [
                "Query 1 for explicitly mentioned medications",
                "Query 2 for explicitly mentioned medications",
                "... (12 queries total, 4 for each medicine parameter, focused on explicitly mentioned medications)"
            ],
            "parameters_analysis": {{
                "Diet Confirmation": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["entity1", "entity2"],
                    "score": 0.0,
                    "justification": "explanation",
                    "strengths": ["strength1", "strength2"],
                    "shortcomings": ["shortcoming if score not perfect"]
                    "improvements": ["specific improvement suggestion based on shortcomings"]
                }},
                "Food Brand Name": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["entity1", "entity2"], 
                    "score": 0.0,
                    "justification": "explanation",
                    "strengths": ["strength1"],
                    "shortcomings": ["shortcoming if score not perfect"]
                    "improvements": ["specific improvement suggestion based on shortcomings"]
                }},
                "Technical Parameters": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["entity1", "entity2"],
                    "score": 0.0,
                    "justification": "explanation", 
                    "strengths": ["strength1"],
                    "shortcomings": ["shortcoming if score not perfect"]
                    "improvements": ["specific improvement suggestion based on shortcomings"]
                }},
                "Treatment Plan and Instructions PSP": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["entity1", "entity2"],
                    "score": 0.0,
                    "justification": "explanation",
                    "strengths": ["strength1"],
                    "shortcomings": ["shortcoming if score not perfect"]
                    "improvements": ["specific improvement suggestion based on shortcomings"]
                }},
                "Medicine Name": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["ONLY explicitly mentioned medication names"],
                    "score": 0.0,
                    "justification": "explanation focusing on explicitly mentioned medications only",
                    "strengths": ["strength1"], 
                    "shortcomings": ["shortcoming if score not perfect"]
                }},
                "Medicine Prescribed": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["prescription processes for explicitly mentioned medications"],
                    "score": 0.0,
                    "justification": "explanation focusing on prescription handling of explicitly mentioned medications",
                    "strengths": ["strength1"],
                    "shortcomings": ["shortcoming if score not perfect"]
                }},
                "Medicine Dosage": {{
                    "segments": ["segment1", "segment2"],
                    "entities": ["dosage info for explicitly mentioned medications only"],
                    "score": 0.0,
                    "justification": "explanation focusing on dosage information for explicitly mentioned medications only",
                    "strengths": ["strength1"],
                    "shortcomings": ["shortcoming if score not perfect"]
                }}
            }},
            "global_summary": "3-5 sentence summary of overall technical assessment quality",
            "medicine_findings_summary": "4-6 sentence summary focused on medicine/pharmacy handling of EXPLICITLY MENTIONED medications only"
        }}
        """

        result = self._call_llm(prompt)
        print("=" * 80)
        print("EXTRACTED INFORMATION FROM MEGA ANALYSIS:")
        print("=" * 80)

        if result:
            # Print explicitly mentioned medications
            medications = result.get("explicitly_mentioned_medications", [])
            print(f"EXPLICITLY MENTIONED MEDICATIONS: {medications}")

            # Print generated knowledge queries
            queries = result.get("global_knowledge_queries", [])
            print(f"\nGENERATED KNOWLEDGE QUERIES ({len(queries)} total):")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query}")

            # Print pet details
            pet_details = result.get("global_pet_details", {})
            print(f"\nEXTRACTED PET DETAILS:")
            for key, value in pet_details.items():
                print(f"  {key}: {value}")

            # Print parameter scores
            params_analysis = result.get("parameters_analysis", {})
            print(f"\nPARAMETER SCORES:")
            for param, analysis in params_analysis.items():
                score = analysis.get("score", 0)
                entities = analysis.get("entities", [])
                print(f"  {param}: Score={score}, Entities={entities}")

        print("=" * 80)

        return result

    def _mega_knowledge_comparison(
        self, analysis_result: Dict, knowledge: List[Dict]
    ) -> Dict:
        """
        SINGLE MEGA LLM CALL: Generate medicine comparisons and knowledge relevance
        for all medicine parameters simultaneously.
        """
        if not knowledge:
            return {
                "medicine_comparisons": {
                    "Medicine Name": "No reference knowledge available for comparison.",
                    "Medicine Prescribed": "No reference knowledge available for comparison.",
                    "Medicine Dosage": "No reference knowledge available for comparison.",
                },
                "knowledge_with_relevance": [],
            }

        # Extract medicine-related segments and entities from analysis
        medicine_data = {}
        for param in self.kb_comparison_params:
            param_analysis = analysis_result.get("parameters_analysis", {}).get(
                param, {}
            )
            medicine_data[param] = {
                "segments": param_analysis.get("segments", []),
                "entities": param_analysis.get("entities", []),
                "score": param_analysis.get("score", 0),
            }

        # Format pet context
        pet_details = analysis_result.get("global_pet_details", {})
        pet_context = ""
        if pet_details:
            pet_context = "Pet Context:\n"
            for key, value in pet_details.items():
                if value and key not in ["conditions", "medications"]:
                    pet_context += f"- {key}: {value}\n"

            if pet_details.get("conditions"):
                pet_context += f"- Conditions: {', '.join(pet_details['conditions'])}\n"
            if pet_details.get("medications"):
                pet_context += (
                    f"- Current medications: {', '.join(pet_details['medications'])}\n"
                )

        # Get explicitly mentioned medications for focused comparison
        explicitly_mentioned_meds = analysis_result.get(
            "explicitly_mentioned_medications", []
        )
        mentioned_meds_text = f"EXPLICITLY MENTIONED MEDICATIONS IN TRANSCRIPT: {', '.join(explicitly_mentioned_meds) if explicitly_mentioned_meds else 'None'}"

        # Format knowledge with numbered sections for easier reference
        knowledge_sections = ""
        for i, k in enumerate(knowledge, 1):
            knowledge_sections += f"\n=== KNOWLEDGE SECTION {i} ===\n{k['text']}\n"

        # Format medicine data with clear structure
        medicine_segments_text = ""
        for param, data in medicine_data.items():
            if data["segments"]:
                medicine_segments_text += f"\n=== {param.upper()} SEGMENTS ===\n"
                for j, segment in enumerate(data["segments"], 1):
                    medicine_segments_text += f"Segment {j}: {segment}\n"
                medicine_segments_text += (
                    f"Extracted Entities: {', '.join(data['entities'])}\n"
                )

        prompt = f"""
        You are a veterinary expert comparing actual consultation practices with evidence-based standards.
        Perform DETAILED medicine comparison analysis focusing on CONCRETE RECOMMENDATIONS from knowledge base.

        {pet_context}
        
        {mentioned_meds_text}

        CRITICAL INSTRUCTIONS FOR MEDICINE COMPARISON:
        1. EXTRACT SPECIFIC MEDICINE NAMES from knowledge base that are relevant to this pet's condition
        2. EXTRACT SPECIFIC DOSAGE INFORMATION from knowledge base for medicines relevant to this pet
        3. COMPARE mentioned medicines/dosages with knowledge-based alternatives
        4. PROVIDE CONCRETE "AS PER STANDARD" recommendations with actual medicine names and dosages from knowledge
        
        KNOWLEDGE BASE SECTIONS TO ANALYZE:
        {knowledge_sections}

        TRANSCRIPT MEDICINE INFORMATION:
        {medicine_segments_text}

        STEP-BY-STEP ANALYSIS PROCESS:

        STEP 1: KNOWLEDGE BASE MEDICINE EXTRACTION
        From each knowledge section, extract:
        - Specific medicine names mentioned for conditions similar to this pet
        - Exact dosage recommendations with units (mg/kg, tablets, ml, etc.)
        - Alternative medicine options for the same conditions
        - Contraindications or safety considerations for this pet type

        STEP 2: TRANSCRIPT MEDICINE EXTRACTION  
        From transcript segments, extract:
        - Exact medicine names mentioned by the veterinarian
        - Specific dosage information provided (amount, frequency, duration)
        - Administration methods discussed
        - Prescription processes mentioned

        STEP 3: DETAILED COMPARISON FOR EACH PARAMETER

        For MEDICINE NAME comparison:
        - MENTIONED: List exact medicine names from transcript
        - AS PER STANDARD: List specific alternative/recommended medicine names from knowledge base for this pet's condition and characteristics
        - GAPS: Compare medicine choices (generic vs branded, alternative options, better suited medicines)

        For MEDICINE DOSAGE comparison:
        - MENTIONED: Extract exact dosage info from transcript (amount, frequency, duration)
        - AS PER STANDARD: Provide specific dosage recommendations from knowledge base for this pet's weight/age/condition
        - GAPS: Compare dosing accuracy (under/over dosing, frequency issues, duration concerns)

        For MEDICINE PRESCRIBED comparison:
        - MENTIONED: Describe prescription process mentioned in transcript
        - AS PER STANDARD: Describe proper prescription protocols from knowledge base
        - GAPS: Identify prescription process shortcomings

        STEP 4: RELEVANCE ASSESSMENT
        For each knowledge section, explain how it specifically relates to the medicines discussed for this pet.

        IMPORTANT FORMATTING REQUIREMENTS:
        - In "MENTIONED" section: Use actual quotes or paraphrases from transcript
        - In "AS PER STANDARD" section: Use specific medicine names, dosages, and recommendations from knowledge base
        - In "GAPS" section: Highlight specific differences with actionable insights
        - Keep each section concise but informative (MENTIONED: max 20 words, AS PER STANDARD: max 35 words, GAPS: max 20 words)

        EXAMPLE FORMAT FOR MEDICINE NAME:
        • MENTIONED: "Brevicto for tick and flea prevention"
        • AS PER STANDARD: "Bravecto, Nexgard, or Simparica Trio recommended for 25kg outdoor dogs with monthly dosing"
        • GAPS: "Brand name correct, could mention alternative options"

        EXAMPLE FORMAT FOR MEDICINE DOSAGE:
        • MENTIONED: "One tablet works for three months"
        • AS PER STANDARD: "20-40kg dogs: 1000mg fluralaner tablet every 12 weeks, monitor for adverse reactions"
        • GAPS: "Duration mentioned but missing weight-specific dosing details"

        Return as JSON:
        {{
            "knowledge_with_relevance": [
                {{
                    "text": "knowledge text",
                    "relevance_explanation": "Specific relevance to medicines discussed for this pet including condition match, dosage applicability, and alternative options"
                }}
            ],
            "medicine_comparisons": {{
                "Medicine Name": "• MENTIONED: [specific medicines from transcript]\\n• AS PER STANDARD: [specific medicine names from knowledge base for this pet]\\n• GAPS: [specific differences in medicine selection]",
                "Medicine Prescribed": "• MENTIONED: [prescription process from transcript]\\n• AS PER STANDARD: [proper prescription protocols from knowledge base]\\n• GAPS: [prescription process improvements needed]",
                "Medicine Dosage": "• MENTIONED: [exact dosage info from transcript]\\n• AS PER STANDARD: [specific dosage recommendations from knowledge base for this pet]\\n• GAPS: [dosing accuracy and completeness issues]"
            }}
        }}
        """

        print("=" * 80)
        print("KNOWLEDGE BASE COMPARISON INPUT:")
        print("=" * 80)
        print(f"NUMBER OF KNOWLEDGE CHUNKS BEING USED: {len(knowledge)}")

        print("\nKNOWLEDGE CHUNKS BEING COMPARED:")
        for i, k in enumerate(knowledge, 1):
            print(f"Chunk {i}: {k['text'][:150]}...")

        print(f"\nMEDICINE DATA BEING COMPARED:")
        for param in self.kb_comparison_params:
            param_analysis = analysis_result.get("parameters_analysis", {}).get(
                param, {}
            )
            entities = param_analysis.get("entities", [])
            score = param_analysis.get("score", 0)
            print(f"  {param}: Entities={entities}, Score={score}")

        print(f"\nEXPLICITLY MENTIONED MEDICATIONS: {explicitly_mentioned_meds}")

        result = self._call_llm(prompt)

        if result:
            comparisons = result.get("medicine_comparisons", {})
            print(f"\nGENERATED MEDICINE COMPARISONS:")
            for param, comparison in comparisons.items():
                print(f"{param}:")
                print(f"  {comparison}")
                print()

        print("=" * 80)

        return result

    def score_technical_assessment(self, transcript: str) -> Dict:
        """
        Score all technical assessment parameters using only 2-3 LLM calls total.

        Args:
            transcript: Call transcript text

        Returns:
            Dictionary with scores for all parameters
        """
        print("=" * 100)
        print("STARTING ULTRA-OPTIMIZED TECHNICAL ASSESSMENT")
        print("=" * 100)
        # CALL 1: Mega comprehensive analysis of all parameters
        print("Making LLM Call 1/3: Mega comprehensive analysis...")
        mega_analysis = self._mega_comprehensive_analysis(transcript)

        if not mega_analysis:
            # Return empty results if analysis failed
            empty_results = {}
            for parameter in self.technical_parameters.keys():
                empty_results[parameter] = {
                    "parameter": parameter,
                    "score": 0,
                    "max_score": self._get_parameter_config(parameter)["max_score"],
                    "percentage": 0,
                    "weighted_score": 0,
                    "weight": self.technical_parameters[parameter],
                    "confidence": "Low",
                    "method": "analysis failed",
                    "strengths": [],
                    "shortcomings": ["Analysis could not be completed."],
                    "segments_analyzed": 0,
                    "entities": [],
                    "comparative_summary": None,
                    "pet_context": None,
                }

            return {
                "category": "Technical Assessment",
                "overall_score": 0,
                "parameters": empty_results,
                "summary": "Technical assessment could not be completed due to analysis failure.",
                "medicine_findings": "Medicine analysis could not be completed.",
            }

        # CALL 2: Knowledge base comparison (if KB available)
        knowledge_comparison = {}
        if self.kb:
            knowledge_queries = mega_analysis.get("global_knowledge_queries", [])
            if knowledge_queries:
                print(
                    "Making LLM Call 2/3: Retrieving knowledge and generating comparisons..."
                )
                knowledge = self._retrieve_knowledge_batch(knowledge_queries)
                if knowledge:
                    knowledge_comparison = self._mega_knowledge_comparison(
                        mega_analysis, knowledge
                    )
                else:
                    print("No knowledge retrieved from database")
            else:
                print("No knowledge queries generated")
        else:
            print("No knowledge base available - skipping knowledge comparison")

        # Process results for all parameters
        results = {}
        total_weighted_score = 0
        total_weight = sum(self.technical_parameters.values())

        parameters_analysis = mega_analysis.get("parameters_analysis", {})
        medicine_comparisons = knowledge_comparison.get("medicine_comparisons", {})

        # Helper function to find matching parameter in analysis results
        def find_matching_param_analysis(target_param_name, analysis_dict):
            """Find parameter analysis with flexible name matching"""
            # Try exact match first
            if target_param_name in analysis_dict:
                return analysis_dict[target_param_name]

            # Try case-insensitive matching
            target_lower = target_param_name.lower()
            for param_name, param_data in analysis_dict.items():
                if param_name.lower() == target_lower:
                    return param_data

            # Try partial matching for common variations
            target_normalized = (
                target_param_name.lower()
                .replace(" and ", " ")
                .replace("psp", "")
                .strip()
            )
            for param_name, param_data in analysis_dict.items():
                param_normalized = (
                    param_name.lower().replace(" and ", " ").replace("psp", "").strip()
                )
                if (
                    target_normalized in param_normalized
                    or param_normalized in target_normalized
                ):
                    return param_data

            return {}

        for parameter, weight in self.technical_parameters.items():
            # Use flexible parameter matching
            param_analysis = find_matching_param_analysis(
                parameter, parameters_analysis
            )
            param_config = self._get_parameter_config(parameter)
            max_score = param_config["max_score"]

            # Extract analysis results
            raw_score = param_analysis.get("score", 0)
            segments = param_analysis.get("segments", [])
            entities = param_analysis.get("entities", [])
            strengths = param_analysis.get("strengths", [])
            shortcomings = param_analysis.get("shortcomings", [])
            improvements = (
                param_analysis.get("improvements", [])
                if parameter not in self.kb_comparison_params
                else []
            )
            raw_percentage = (
                raw_score / 5
            ) * 100  # LLM always scores 0-5, convert to percentage
            normalized_score = (raw_percentage / 100) * max_score

            weighted_score = (raw_percentage / 100) * weight * 100

            # Debug print to see what's happening
            if parameter == "Treatment Plan and Instructions PSP":
                print(f"DEBUG - Parameter: {parameter}")
                print(f"DEBUG - Found analysis: {param_analysis}")
                print(f"DEBUG - Raw score: {raw_score}")
                print(f"DEBUG - Segments: {len(segments)}")
                print(f"DEBUG - Entities: {entities}")

            # Get comparative summary if available
            comparative_summary = (
                medicine_comparisons.get(parameter)
                if parameter in self.kb_comparison_params
                else None
            )

            # Determine method and confidence
            method = (
                "lenient pharmacy assessment"
                if parameter in self.pharmacy_params
                else "direct LLM assessment"
            )
            confidence = (
                "High"
                if len(segments) >= 2 and raw_score > 3.5
                else "Medium" if len(segments) >= 1 and raw_score > 2.0 else "Low"
            )

            results[parameter] = {
                "parameter": parameter,
                "score": normalized_score,
                "max_score": max_score,
                "percentage": raw_percentage,
                "weighted_score": weighted_score,
                "weight": weight,
                "confidence": confidence,
                "method": method,
                "strengths": strengths,
                "shortcomings": (
                    shortcomings
                    if shortcomings
                    else ["No specific shortcomings identified."]
                ),
                "improvements": (
                    improvements if parameter not in self.kb_comparison_params else []
                ),
                "segments_analyzed": len(segments),
                "entities": entities,
                "comparative_summary": comparative_summary,
                "pet_context": mega_analysis.get("global_pet_details"),
            }

            total_weighted_score += weighted_score

        # Calculate overall category score
        total_max_score = sum(
            param_config["max_score"]
            for param_config in [
                self._get_parameter_config(p) for p in self.technical_parameters.keys()
            ]
        )
        actual_total_score = sum(result["score"] for result in results.values())
        category_percentage = (
            (actual_total_score / total_max_score) * 100 if total_max_score > 0 else 0
        )
        # Get summaries from mega analysis
        global_summary = mega_analysis.get(
            "global_summary", "Technical assessment completed."
        )
        medicine_summary = mega_analysis.get(
            "medicine_findings_summary", "Medicine analysis completed."
        )

        print(f"Technical Assessment completed with {2 if self.kb else 1} LLM calls.")

        return {
            "category": "Technical Assessment",
            "total_score": round(actual_total_score, 1),
            "max_possible_score": total_max_score,
            "overall_score": round(category_percentage, 2),
            "parameters": results,
            "summary": global_summary,
            "medicine_findings": medicine_summary,
            "llm_calls_used": 2 if self.kb else 1,  # Track actual calls made
        }


def main():
    """
    Example usage of the UltraOptimizedTechnicalAssessmentScorer.
    """
    import os
    from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY

    # Initialize knowledge base if available
    try:
        from knowledge_base import CloudVeterinaryKnowledgeBase

        kb = CloudVeterinaryKnowledgeBase(
            collection_name=COLLECTION_NAME,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
        )
        print("Knowledge base initialized successfully")
    except Exception as e:
        print(f"Could not initialize knowledge base: {e}")
        kb = None

    # Get OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize ultra-optimized scorer
    scorer = UltraOptimizedTechnicalAssessmentScorer(
        knowledge_base=kb, openai_api_key=openai_api_key
    )

    # Example transcript
    transcript = """
   # Diarized Transcript

File: mehak_nazar_supertails_in__Pharmacy_OB__320__-1__7989796312__2025-05-04_19-54-19.mp3 (Duration: 9:18)

Dr. Mehek : Hello.

Kranthi Chaitanya : Hello. Hi, good evening.

Dr. Mehek : Ah, good evening. Am I speaking to Kranthi Chaitanya?

Kranthi Chaitanya : Yes ma'am, tell me.

Dr. Mehek : I am Dr. Mehek calling from Supertails.

Kranthi Chaitanya : Okay.

Dr. Mehek : And this is in regard to pharmacy verification for the order you have placed on our platform. Could you please confirm the order?

Kranthi Chaitanya : Yes ma'am.

Dr. Mehek : So, I can see that you have placed an order for one Samparica tablet and 20 kg body weight.

Kranthi Chaitanya : Yes, one endofloxacin 50 mg tablet and one Alembic mextin tablet.

Dr. Mehek : Right. So, may I know what is the issue present with your pet right now?

Kranthi Chaitanya : What, what are you asking?

Dr. Mehek : May I know what is the issue present with your pet right now?

Kranthi Chaitanya : Ticks, please.

Dr. Mehek : Okay.

Kranthi Chaitanya : It was not controlling by any medication or injections. That's why someone suggested that medicine.

Dr. Mehek : Okay. Can you share the pet details? What is the name of the pet?

Kranthi Chaitanya : Lucky.

Dr. Mehek : Okay. Can you share the pet details? What is the name of the pet?

Kranthi Chaitanya : Lucky.

Kranthi Chaitanya : It's about two years.

Dr. Mehek : Two years. Okay. Which breed?

Kranthi Chaitanya : Male.

Dr. Mehek : Breed?

Kranthi Chaitanya : Lab.

Dr. Mehek : So right now, what kind of food you are feeding him? Homemade or some?

Kranthi Chaitanya : Homemade and Pedigree.

Dr. Mehek : Okay. And what would be body weight for him right now?

Kranthi Chaitanya : 10 to 20 between.

Dr. Mehek : Okay.

Kranthi Chaitanya : Only homemade and Pedigree.

Dr. Mehek : Okay. And what would be body weight for him right now?

Kranthi Chaitanya : 10 to 20 between.

Dr. Mehek : Okay.

Dr. Mehek : So, I will be sending you this prescription right away on this number in the form of the link. Are there any other questions or any other health related concerns regarding Lucky which you want to discuss right now?

Kranthi Chaitanya : We already used so many medicines to control those ticks, but it was not controlling and it was like hair fall and became thin.

Dr. Mehek : Okay. Food digestion is okay? Food eating is okay? Digestion is okay, but hair fall and the ticks are not controlling?

Dr. Mehek : So, where exactly is it affecting? Like are there some patches?

Kranthi Chaitanya : No, no patches. Only ticks are there. It was fell down and hair loss is more. Hair loss is more.

Dr. Mehek : All over the body or some particular regions?

Kranthi Chaitanya : All over the body, mainly underarms, tail and the ears, neck, belt area.

Dr. Mehek : Okay. Fine.

Kranthi Chaitanya : Itching is also there.

Dr. Mehek : Itching is there when it is irritated by ticks?

Kranthi Chaitanya : Itching is there but not more than that.

Dr. Mehek : Okay. Itching is there when it is irritated by ticks.

Dr. Mehek : Okay fine. Itching is there when it is irritated by ticks, otherwise not there?

Kranthi Chaitanya : Not there.

Dr. Mehek : Okay.

Kranthi Chaitanya : Since from 2 to 3 months suffering.

Dr. Mehek : Okay. Okay. So what else you had treated before?

Kranthi Chaitanya : With antibiotic tablet and some injections to control and spot on powder, shampoos and some chemical spray also we used.

Dr. Mehek : Okay. Okay. I don't know the names. Recently they brought to us.

Dr. Mehek : Okay. Okay. So right now I want to take care of the dog.

Dr. Mehek : Okay.

Dr. Mehek : I don't know the names. Recently they brought to us.

Dr. Mehek : So right now I want to take care of the dog.

Dr. Mehek : Yeah, you can continue with this tablet if ticks are there. This tablet is okay for that and this also you can give. And this antibiotic is not needed.

Dr. Mehek : Yeah, you can keep it for like future use, but it is not needed as such.

Dr. Mehek : You can share the pictures of the dog on the WhatsApp number so that I can see.

Kranthi Chaitanya : Ticks are coming out only for two days and later than two days they are not outside and again they are coming for two days.

Dr. Mehek : Oh, like that it is happening. When they become big, they fall down and they are dead.

Kranthi Chaitanya : Yeah.

Dr. Mehek : Then you can give this tablet. Simparica tablet is sufficient and Ivermectin also one dose you can give.

Dr. Mehek : And deworming also we are giving every month?

Kranthi Chaitanya : Not month. It has to be done after every three months.

Dr. Mehek : Every three months?

Kranthi Chaitanya : Yes, yes. Every month is not required.

Dr. Mehek : Okay. Okay. They told me that give deworming first.

Kranthi Chaitanya : Yes, yes. That is good you can do deworming, but it has to be done every three months. Now if you are done now, we can do it after three months.

Dr. Mehek : Okay okay.

Kranthi Chaitanya : Okay.

Kranthi Chaitanya : Yes, every month is not required.

Dr. Mehek : Okay. Okay. They told me that give deworming first.

Kranthi Chaitanya : Yes, yes, that is good you can do deworming, but it has to be done every three months. Now if you are done now, we can do it after three months.

Dr. Mehek : Okay okay.

Kranthi Chaitanya : Okay.

Dr. Mehek : Later I can buy from your site?

Kranthi Chaitanya : Yes, yes, you can purchase it from here only.

Dr. Mehek : What about the ticks, the tablet doses for them?

Kranthi Chaitanya : One complete tablet you are supposed to give. That Simparica one, that 10 to 20 you are supposed to give it one complete. And for the Ivermectin that another tablet you have purchased, that you are supposed to give half tablet. Give it once only.

Dr. Mehek : Okay. Half tablet. For how many days I have to give Simparica?

Kranthi Chaitanya : Simparica only once it is supposed to be given. After every month it is supposed to be repeated.

Dr. Mehek : Okay, one tablet for one day, only for one month?

Kranthi Chaitanya : One month. Yes, yes.

Dr. Mehek : Okay, half tablet, that another tablet?

Kranthi Chaitanya : Another. Yeah. That you give first one and then after one week you can give another dose, another half.

Dr. Mehek : Okay. That tablet have to give one week half tablet one week?

Kranthi Chaitanya : Yes, yes.

Dr. Mehek : Okay okay. Okay. My dog will eat Idli. We are South Indians. I will keep tablet in it and he will eat completely.

Kranthi Chaitanya : Yes, yes, you can give that.

Dr. Mehek : That tablet have to give one week half tablet one week?

Kranthi Chaitanya : Yes, yes.

Dr. Mehek : Okay okay. Okay. My dog will eat Idli. We are South Indians. I will keep tablet in it and he will eat completely.

Kranthi Chaitanya : Yes, yes, you can give. That is not a problem.

Dr. Mehek : No, no problem. You can mix. We are keeping chicken legs only with turmeric water boiling only.

Kranthi Chaitanya : Okay.

Dr. Mehek : No masala, no added anything.

Kranthi Chaitanya : Okay okay.

Dr. Mehek : Not even salt also.

Kranthi Chaitanya : Okay okay.

Dr. Mehek : And pedigree I am giving pedigree.

Kranthi Chaitanya : Yes, yes.

Dr. Mehek : Is there any other food?

Kranthi Chaitanya : No, this is sufficient, you can give this. This is sufficient right now.

Dr. Mehek : Okay and next we are, this is summer right? We are giving buttermilk in afternoon.

Kranthi Chaitanya : Buttermilk yeah, that is better.

Dr. Mehek : Lightly without adding anything, buttermilk twice a day.

Kranthi Chaitanya : Buttermilk, yeah that is better.

Dr. Mehek : Lightly without adding anything, buttermilk twice a day and normal rice food. That's it.

Kranthi Chaitanya : Sufficient, it is sufficient. You can add some pumpkin also, carrots or some vegetables. That also.

Dr. Mehek : Sweet potato he will eat.

Kranthi Chaitanya : Sweet potato, watermelon, apple, papaya, fruits also.

Dr. Mehek : Then we are eating if we give he will eat.

Kranthi Chaitanya : He will also eat. Yeah, yeah, no issue. You can give him anything.

Dr. Mehek : Okay, okay. Then one thing I want to ask. Our dog is so friendly.

Kranthi Chaitanya : Okay.

Dr. Mehek : Even outsiders come also he won't bark.

Kranthi Chaitanya : Oh okay.

Dr. Mehek : Then one thing I want to ask. Our dog is so friendly.

Kranthi Chaitanya : Okay.

Dr. Mehek : Even outsiders come also he won't bark.

Kranthi Chaitanya : Oh okay.

Dr. Mehek : He will be friendly going to them. That was the complaint they gave to us.

Kranthi Chaitanya : Okay, okay, that is okay. That is good only that he is friendly.

Dr. Mehek : For security purpose only, outsider people will maintain dog.

Kranthi Chaitanya : He is so friendly and he will allow all outsiders also.

Dr. Mehek : He won't bark even when he's hungry, only barks then.

Kranthi Chaitanya : Oh okay, okay.

Dr. Mehek : In the early six to seven he has to get pedigree. Otherwise he will bark.

Kranthi Chaitanya : Oh okay, okay.

Dr. Mehek : If any other outsiders come he won't bark. Just giving shake hands to all of them.

Dr. Mehek : His color is black. Looks like devil but acts like child.

Kranthi Chaitanya : Okay, okay.

Dr. Mehek : That is the main reason they given us.

Kranthi Chaitanya : Okay, okay.

Dr. Mehek : Okay ma'am. If any problem I can message in the WhatsApp.

Kranthi Chaitanya : Yeah, yeah, sure sure. You can always reach out to us. We chat with vet option.

Kranthi Chaitanya : Okay. Anytime schedule consultation.

Dr. Mehek : Okay, okay.

Kranthi Chaitanya : Okay. Thank you. Bye. Have a good evening.

Dr. Mehek : Thank you ma'am.


---
Transcript processed on Sat May 10 10:55:39 2025
    """

    # Score the transcript
    print("Starting ultra-optimized technical assessment...")
    scores = scorer.score_technical_assessment(transcript)

    # Print results
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
