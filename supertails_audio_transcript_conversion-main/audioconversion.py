#!/usr/bin/env python3
"""
Audio Transcription and Speaker Diarization Script
Converts audio files to text with speaker identification using OpenAI Whisper and NeMo
"""

import torch
import numpy as np
import librosa
import nemo.collections.asr as nemo_asr
import json
import os
from datetime import datetime
import time
import torchaudio
from openai import OpenAI
import anthropic
from omegaconf import OmegaConf
import math
import glob
import soundfile as sf
from dotenv import load_dotenv
import shutil

# Import alternative diarization as fallback
try:
    from alternative_diarization import get_speaker_segments

    ALTERNATIVE_DIARIZATION_AVAILABLE = True
except ImportError:
    ALTERNATIVE_DIARIZATION_AVAILABLE = False
    print("Warning: Alternative diarization not available")

load_dotenv()  # Load environment variables from .env file


def enhance_transcript_with_claude(
    timestamped_words, speaker_segments, english_translation, output_file_path, mp3_file
):
    """
    Use Claude to enhance transcript by:
    1. Properly attributing words to speakers based on timing and context
    2. Adding punctuation and fixing grammatical issues
    3. Removing duplicates, filler words, and repetitions
    4. Translating ALL content to English

    Args:
        timestamped_words: List of words with timestamps
        speaker_segments: List of speaker segments from diarization
        english_translation: Direct English translation from Whisper
        output_file_path: Path to save the enhanced transcript
        mp3_file: Original audio file path

    Returns:
        Enhanced transcript with proper speaker attribution in English
    """

    # Get API key from environment variable
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=claude_api_key)

    # Print debug info
    print(
        f"Processing transcript with {len(timestamped_words)} words and {len(speaker_segments)} speaker segments"
    )

    # Split processing based on speaker_segments to create manageable chunks
    chunk_size = 20
    overlap_size = 5  # Number of segments to overlap
    num_segments = len(speaker_segments)

    # Adjust the calculation of chunks to account for overlap
    adjusted_step = chunk_size - overlap_size
    num_chunks = max(
        1, (num_segments - overlap_size + adjusted_step - 1) // adjusted_step
    )

    all_enhanced_segments = []
    previous_chunk_result = None  # Store the previous chunk's result

    for chunk_idx in range(num_chunks):
        # Calculate start and end indices with overlap
        start_idx = max(0, chunk_idx * adjusted_step)
        end_idx = min(start_idx + chunk_size, num_segments)

        print(f"\nDEBUG: start_idx : {start_idx}")
        print(f"DEBUG: end_idx : {end_idx}")

        # If we've processed all segments, break
        if start_idx >= num_segments:
            break

        # Extract the speaker segments for this chunk
        chunk_speaker_segments = speaker_segments[start_idx:end_idx]

        # Find start and end times for this chunk
        chunk_start_time = chunk_speaker_segments[0]["start"]
        chunk_end_time = chunk_speaker_segments[-1]["end"]

        print(f"\nDEBUG: chunk_start_time : {chunk_start_time}")
        print(f"DEBUG: chunk_end_time : {chunk_end_time}")

        # Extract words that fall within this time range
        chunk_words = [
            word
            for word in timestamped_words
            if float(word["start"]) >= chunk_start_time
            and float(word["end"]) <= chunk_end_time
        ]

        # Create timestamps to help identify the current section in the translation
        timestamp_markers = f"This chunk covers approximately timestamps from {chunk_start_time:.2f}s to {chunk_end_time:.2f}s"

        # Add context from previous chunk if available
        previous_context = ""
        if previous_chunk_result and chunk_idx > 0:
            # Take just the overlapping segments from the previous chunk to provide context
            overlap_context = previous_chunk_result[-overlap_size:]
            previous_context = f"""
            IMPORTANT CONTEXT:
            The following is the COMPLETE result from processing the previous chunk:
            Use this to maintain consistency in speaker names and avoid duplicating content:

            {json.dumps(previous_chunk_result, indent=2)}

            When processing this chunk, if you encounter content that appears to be the same as what's shown above,
            do NOT include it in your output as it has already been processed.
            Focus on transcribing ONLY the new content from this chunk while maintaining speaker consistency.

            Your task is to SEAMLESSLY CONTINUE the transcript where the previous result ended,
            with no duplications or omissions
            """

        # Prepare prompt for Claude - send the complete data
        prompt = f"""
        I have a timestamped words file called timestamped_words and
        speaker segments file called speaker_segments that may contain
        MULTIPLE LANGUAGES and a complete English translation of an audio.

        {timestamp_markers}

        {previous_context}

        I need you to:
        1. Assign each timestamped word to the correct speaker based on timing and context
        2. Add appropriate punctuation
        3. Remove duplicates, repetitions, and filler words
        4. TRANSLATE ALL CONTENT TO ENGLISH - this is a MANDATORY REQUIREMENT
        5. Create a readable conversation

        EXTREMELY IMPORTANT:
        1.You MUST preserve and translate EVERY SINGLE exchange in the conversation,
          regardless of language. This includes:
          - ALL questions and answers, even brief ones
          - ALL acknowledgments (Okay, Yes, Yeah, etc.)
          - ALL speaker turns, even if they seem repetitive
          - ALL information provided by either speaker in ANY language

        2. You MUST use the actual speaker names found in the full translated text instead of
        generic labels like "Speaker 1" or "Speaker 2"

        3. You MUST TRANSLATE ALL NON-ENGLISH text to English
          - Even if you're uncertain about the exact translation, provide your best attempt
          - NEVER omit content because it's in a foreign language
          - NEVER skip any dialogue exchanges in any language
          - COMPLETENESS is more important than perfect translation

        4. If a company name appears in the conversation, you MUST use the correct name "Supertails"
           - Ensure consistent and correct spelling of "Supertails" throughout
           - This is the official company name and must be preserved exactly

        Do NOT omit any dialogue exchanges. The only things you should remove are:
          - Word repetitions that occur within the same utterance ("I I I think")

        Here are ALL the timestamped words with their timing information:
        {json.dumps(chunk_words, indent=2)}

        Here are ALL the speaker segments with timing:
        {json.dumps(chunk_speaker_segments, indent=2)}

        Here is the full translated text (for context):
        {english_translation}

        Format the response as JSON with this structure:
        [
          {{"speaker": "Speaker X", "text": "Complete sentence..."}},
          {{"speaker": "Speaker Y", "text": "Another complete sentence..."}}
        ]

        FINAL CHECK: Before submitting your response, perform these verification steps:
        1. Count the number of dialogue turns in your response
        2. Ensure it matches approximately the number of turns in the speaker_segments, english_translation
        3. Verify that EVERY exchange from the english_translation that falls in this time range has been included in your response
        and aligned with the appropriate speaker.
        4. Verify all texts barring the Word repetitions as present in the english_translation that falls in this time range
        have been included in the response and aligned with the appropriate speaker by applying your reasoning.
        5. If you find you've omitted anything, add it back in the appropriate place
        6. Use "ma'am" or "sir" rather than informal terms like "mom" or "dad"

        I will measure your performance by how completely you preserve ALL dialogue
        exchanges while translating to English.
        """

        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0.1,
            system="""You are an expert in multilingual audio transcription processing
            and translation. Your company name is "Supertails". Your task is to clean up transcripts while preserving
            and translating EVERY SINGLE dialogue exchange in ALL languages. Never
            omit any exchanges, questions, or responses, no matter how brief or what
            language they are in, UNLESS they have already been included in a previous chunk.
            Your primary goal is COMPLETE preservation of all NEW dialogue content while
            translating everything to English, with ZERO DUPLICATION of previously processed content.
            YOU MUST RESPOND WITH ONLY VALID JSON WITHOUT ANY EXPLANATORY TEXT.
            YOUR ENTIRE RESPONSE MUST BEGIN WITH '[' AND END WITH ']' - DO NOT ADD ANY OTHER TEXT.""",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON from response
        response_text = message.content[0].text
        print(f"\n\nDEBUG: response_text : {response_text}")
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1

        json_text = response_text[json_start:json_end]
        enhanced_segments = json.loads(json_text)

        # Store the current result for the next iteration
        previous_chunk_result = enhanced_segments

        # Add this chunk's results to the accumulated results
        all_enhanced_segments.extend(enhanced_segments)
        print(
            f"Processed chunk {chunk_idx+1}/{num_chunks}: added {len(enhanced_segments)} segments"
        )

    # Write to text file if path is provided
    if output_file_path:
        audio_filename = os.path.basename(mp3_file)
        audio_filename_no_ext = os.path.splitext(audio_filename)[0]
        timestamp = str(int(time.time()))
        diarization_filename = f"{audio_filename_no_ext}_diarization_{timestamp}.txt"
        # Always use /tmp/output for Lambda compatibility
        diarization_dir = "/tmp/output"
        os.makedirs(diarization_dir, exist_ok=True)
        diarization_filepath = os.path.join(diarization_dir, diarization_filename)
        with open(diarization_filepath, "w", encoding="utf-8") as f:
            # Get duration from the audio file
            speech_array, sampling_rate = librosa.load(mp3_file, sr=16000)
            duration_seconds = len(speech_array) / sampling_rate
            duration_formatted = (
                f"{int(duration_seconds//60)}:{int(duration_seconds%60):02d}"
            )

            # Write header
            f.write("# Diarized Transcript\n\n")
            f.write(f"File: {audio_filename} (Duration: {duration_formatted})\n\n")

            # Write each dialogue turn
            for segment in all_enhanced_segments:
                f.write(f"{segment['speaker']} : {segment['text']}\n\n")

            # Write footer with simple timestamp
            f.write(f"\n---\nTranscript processed on {time.ctime()}")

        print(f"Diarized Transcript saved to {diarization_filepath}")

    print(
        f"Claude enhanced transcript successfully with {len(all_enhanced_segments)} segments"
    )
    return all_enhanced_segments


def transcribe_and_diarize_mp3_two_speakers(audio_path, diarization_path):
    """
    Transcribe MP3 audio using Whisper and perform speaker diarization using NeMo.
    Optimized specifically for files with 2 speakers using proper timestamp alignment.
    """
    # Check if NumPy version is 2.0 or higher and np.sctypes is not available
    if not hasattr(np, "sctypes"):
        # Create a monkey patch for np.sctypes
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
        }
        print("Added compatibility layer for NumPy 2.0+ (np.sctypes)")

    print(f"Processing MP3 file: {audio_path}")

    # Step 1: Load MP3 audio file
    print("Loading MP3 file...")
    try:
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        print(f"Audio loaded: {len(speech_array)/sampling_rate:.2f} seconds")
        duration_seconds = len(speech_array) / sampling_rate
        print(f"Audio loaded: {duration_seconds:.2f} seconds")

    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Step 2: Set up speaker diarization using NeMo FIRST
    print("Setting up speaker diarization...")

    # Create a temporary WAV file for NeMo
    temp_audio_path = "/tmp/temp_audio.wav"
    try:
        sf.write(temp_audio_path, speech_array, sampling_rate)
    except ImportError:
        from scipy.io import wavfile

        wavfile.write(
            temp_audio_path, sampling_rate, (speech_array * 32767).astype(np.int16)
        )

    # Create output directory in /tmp
    diar_output_dir = "/tmp/diar_output"
    os.makedirs(diar_output_dir, exist_ok=True)

    # Step 3: Perform speaker diarization using NeMo's ClusteringDiarizer
    print("Running speaker diarization using NeMo (assuming exactly 2 speakers)...")

    try:
        # Create a manifest file for the audio in /tmp/diar_output
        manifest_file = os.path.join(diar_output_dir, "manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(
                {
                    "audio_filepath": temp_audio_path,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": 2,
                },
                f,
            )

        # Create a diarizer config
        diarizer_conf = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "num_workers": 0,
            "batch_size": 1,
            "sample_rate": 16000,
            "verbose": True,
            "diarizer": {
                "manifest_filepath": manifest_file,
                "out_dir": diar_output_dir,
                "oracle_vad": False,
                "collar": 0.0,
                "ignore_overlap": False,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "onset": 0.4,
                        "offset": 0.4,
                        "pad_offset": 0.2,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.3,
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.7,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": 1.5,
                        "shift_length_in_sec": 0.25,
                        "multiscale_weights": [1.0],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "algorithm": "spectral",
                    "parameters": {
                        "affinity": "cosine",
                        "num_speakers": 2,  # Fixed at 2 speakers
                        "oracle_num_speakers": 2,
                        "max_num_speakers": 2,
                        "enhanced_count_thres": 40,
                        "max_rp_threshold": 0.15,
                        "sparse_search_volume": 30,
                    },
                },
                "msdd_model": {"parameters": {"threshold": 0.6, "step_length": 0.25}},
            },
        }

        # Initialize the diarizer
        diarizer_conf = OmegaConf.create(diarizer_conf)

        # Create and run the diarizer
        diarizer = nemo_asr.models.ClusteringDiarizer(cfg=diarizer_conf)
        diarizer.diarize()

        # Load the generated RTTM file
        rttm_file = os.path.join(
            diar_output_dir,
            "pred_rttms",
            os.path.basename(temp_audio_path).replace(".wav", ".rttm"),
        )

        if not os.path.exists(rttm_file):
            raise FileNotFoundError(
                f"Expected RTTM file not found: {rttm_file}. Check: {os.listdir(diar_output_dir)}"
            )

        # Parse RTTM to get speaker segments
        speaker_segments = []
        with open(rttm_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 10:
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    end_time = start_time + duration
                    speaker = parts[7]
                    speaker_segments.append(
                        {"start": start_time, "end": end_time, "speaker": speaker}
                    )

        print(f"DEBUG: RTTM file path: {rttm_file}")
        with open(rttm_file, "r") as f:
            rttm_content = f.read()
        print(
            f"DEBUG: RTTM content length: {len(rttm_content)} chars, {len(rttm_content.splitlines())} lines"
        )
        print(f"DEBUG: First few RTTM lines: {rttm_content.splitlines()[:2]}")

        print(f"NeMo diarization complete. Found {len(speaker_segments)} segments.")

    except Exception as e:
        print(f"NeMo diarization error: {e}")
        print("Using fallback approach...")

        # Try alternative diarization if available
        if ALTERNATIVE_DIARIZATION_AVAILABLE:
            print("Using alternative diarization approach...")
            try:
                speaker_segments = get_speaker_segments(temp_audio_path, num_speakers=2)
                print(
                    f"Alternative diarization complete. Found {len(speaker_segments)} segments."
                )
            except Exception as alt_e:
                print(f"Alternative diarization also failed: {alt_e}")
                # Fallback to a simple split if all diarization fails
                total_duration = len(speech_array) / sampling_rate
                mid_point = total_duration / 2

                speaker_segments = [
                    {"start": 0, "end": mid_point, "speaker": "SPEAKER_0"},
                    {"start": mid_point, "end": total_duration, "speaker": "SPEAKER_1"},
                ]
        else:
            print("Alternative diarization not available, using simple split")
            # Fallback to a simple split if diarization fails
            total_duration = len(speech_array) / sampling_rate
            mid_point = total_duration / 2

            speaker_segments = [
                {"start": 0, "end": mid_point, "speaker": "SPEAKER_0"},
                {"start": mid_point, "end": total_duration, "speaker": "SPEAKER_1"},
            ]

    # Step 4: Initialize OpenAI client
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=openai_api_key)

    # Step 5: Transcribe with OpenAI Whisper API to get word-level timestamps
    print("Transcribing with OpenAI Whisper API to get word-level timestamps...")

    # Check file size and implement chunking if needed
    file_size = os.path.getsize(temp_audio_path)
    max_chunk_size = 24 * 1024 * 1024  # 24MB is OpenAI's limit

    if file_size > max_chunk_size:
        print(
            f"Audio file is {file_size/1024/1024:.2f}MB, exceeds OpenAI's limit. Using chunking."
        )

        # Load audio for chunking
        y, sr = librosa.load(temp_audio_path, sr=16000)

        # Calculate total duration and chunk duration (3 minutes per chunk)
        total_duration = len(y) / sr
        chunk_duration = 180  # 3 minutes in seconds
        num_chunks = math.ceil(total_duration / chunk_duration)

        print(
            f"Splitting audio into {num_chunks} chunks of {chunk_duration} seconds each"
        )

        all_timestamped_words = []
        all_transcription_text = ""

        for i in range(num_chunks):
            # Calculate chunk boundaries
            start_sample = int(i * chunk_duration * sr)
            end_sample = int(min((i + 1) * chunk_duration * sr, len(y)))

            # Extract chunk
            chunk = y[start_sample:end_sample]

            # Save chunk to temporary file in /tmp
            chunk_path = f"/tmp/chunk_{i}.wav"
            sf.write(chunk_path, chunk, sr)

            try:
                # Process chunk with OpenAI
                with open(chunk_path, "rb") as chunk_file:
                    # Call OpenAI Whisper API with word-level timestamps for this chunk
                    chunk_response = client.audio.transcriptions.create(
                        file=chunk_file,
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                    )

                # Adjust timestamps to account for chunk position
                time_offset = i * chunk_duration
                chunk_words = []

                for word_data in chunk_response.words:
                    chunk_words.append(
                        {
                            "text": word_data.word,
                            "start": word_data.start + time_offset,
                            "end": word_data.end + time_offset,
                        }
                    )

                all_timestamped_words.extend(chunk_words)
                all_transcription_text += chunk_response.text + " "

                print(f"Chunk {i+1}/{num_chunks} processed: {len(chunk_words)} words")

                # Clean up chunk file
                os.remove(chunk_path)

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")

        timestamped_words = all_timestamped_words
        transcription = all_transcription_text.strip()

    else:
        # Process the whole file if it's small enough
        with open(temp_audio_path, "rb") as temp_audio_file:
            # Call OpenAI Whisper API with word-level timestamps
            openai_response = client.audio.transcriptions.create(
                file=temp_audio_file, model="whisper-1"
            )

        # Convert OpenAI word timestamps to our format
        timestamped_words = []
        if hasattr(openai_response, "words") and openai_response.words:
            for word_data in openai_response.words:
                timestamped_words.append(
                    {
                        "text": word_data.word,
                        "start": word_data.start,
                        "end": word_data.end,
                    }
                )
        else:
            print("Warning: No word-level timestamps available from OpenAI response")
            # Fallback: create dummy timestamps
            words = openai_response.text.split()
            duration_per_word = (
                len(speech_array) / sampling_rate / len(words) if words else 1
            )
            for i, word in enumerate(words):
                start_time = i * duration_per_word
                end_time = (i + 1) * duration_per_word
                timestamped_words.append(
                    {"text": word, "start": start_time, "end": end_time}
                )

        transcription = openai_response.text

    print(f"DEBUG: OpenAI Whisper API response received")
    print(f"DEBUG: Number of words with timestamps: {len(timestamped_words)}")

    # Debug output similar to the original code
    print(f"DEBUG: Extracted {len(timestamped_words)} timestamped words")
    if timestamped_words:
        print(f"DEBUG: First 15 timestamped words: {timestamped_words[:15]}")
    else:
        print("DEBUG: No timestamped words were extracted")

    # Get English translation directly from the audio
    print("Getting dedicated English translation from the audio...")
    if file_size > max_chunk_size:
        print(f"Using chunking for translation...")

        # We already have chunks created, or we can use the same chunking logic
        english_translation_full = ""

        # Load audio for chunking if not already done
        y, sr = librosa.load(audio_path, sr=16000)

        # Calculate total duration and chunk duration (3 minutes per chunk)
        total_duration = len(y) / sr
        chunk_duration = 180  # 3 minutes in seconds
        num_chunks = math.ceil(total_duration / chunk_duration)

        for i in range(num_chunks):
            # Calculate chunk boundaries
            start_sample = int(i * chunk_duration * sr)
            end_sample = int(min((i + 1) * chunk_duration * sr, len(y)))

            # Extract chunk
            chunk = y[start_sample:end_sample]

            # Save chunk to temporary file in /tmp
            chunk_path = f"/tmp/chunk_{i}.wav"
            sf.write(chunk_path, chunk, sr)

            try:
                # Process chunk with OpenAI for translation
                with open(chunk_path, "rb") as chunk_file:
                    # Call OpenAI Whisper API specifically for translation
                    chunk_translation = client.audio.translations.create(
                        file=chunk_file, model="whisper-1"
                    )

                english_translation_full += chunk_translation.text + " "
                print(f"Translation chunk {i+1}/{num_chunks} processed")

                # Clean up chunk file
                os.remove(chunk_path)

            except Exception as e:
                print(f"Error processing translation chunk {i}: {e}")

        english_translation = english_translation_full.strip()

    else:
        # Process the whole file if it's small enough
        with open(temp_audio_path, "rb") as temp_audio_file:
            # Call OpenAI Whisper API with word-level timestamps
            translation_response = client.audio.transcriptions.create(
                file=temp_audio_file, model="whisper-1"
            )
        english_translation = translation_response.text

    print(
        f"DEBUG: English translation obtained (length: {len(english_translation)} chars)"
    )

    # Step 6: Process diarization results
    print("Processing diarization results...")
    print(f"Number of speaker segments from NeMo: {len(speaker_segments)}")

    # Debug output
    if speaker_segments:
        print(f"First 15 speaker segment: {speaker_segments[0:15]}")
        print(f"Last 15 speaker segment: {speaker_segments[-15:]}")
    else:
        print("WARNING: No speaker segments found from NeMo")
        # Create a dummy speaker segment as fallback
        speaker_segments = [
            {"start": 0, "end": 100000, "speaker": "Speaker 1"}  # Cover entire audio
        ]

    print("Enhancing transcript with Claude...")
    final_segments = enhance_transcript_with_claude(
        timestamped_words,
        speaker_segments,
        english_translation,
        "/tmp/output",
        audio_path,
    )

    # Get unique speaker names from the processed segments
    unique_speakers = sorted(
        list(set(segment["speaker"] for segment in final_segments))
    )

    # Always define transcription_filepath, even if something failed
    transcription_filepath = None
    try:
        audio_filename = os.path.basename(audio_path)
        audio_filename_no_ext = os.path.splitext(audio_filename)[0]
        transcription_filename = f"{audio_filename_no_ext}.txt"
        transcription_filepath = os.path.join("/tmp/output", transcription_filename)
        # Only write transcript if we have a translation and duration
        if english_translation and duration_seconds is not None:
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            duration_formatted = f"{minutes}:{seconds:02d}"
            with open(transcription_filepath, "w", encoding="utf-8") as f:
                f.write(f"File: {audio_filename} (Duration: {duration_formatted})\n\n")
                f.write(english_translation)
            print(f"Transcription saved to: {transcription_filepath}")
        else:
            transcription_filepath = None
    except Exception as e:
        print(f"Failed to save plain transcript: {e}")
        transcription_filepath = None

    # Step 7: Create final output
    results = {
        "audio_path": audio_path,
        "full_transcription": english_translation,
        "speakers": unique_speakers,
        "segments": final_segments,
        "duration_seconds": duration_seconds,
        "processing_info": {
            "whisper_model": "whisper-large-v2",
            "diarization_approach": "nemo_clustering_diarizer",
            "processing_date": datetime.now().isoformat(),
            "num_speakers": len(unique_speakers),
            "enhanced_with_claude": True,
        },
        "transcription_file": transcription_filepath,
    }

    # IMPORTANT: Clean up temporary files ONLY at the very end
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
        print(f"Cleaned up temporary file: {temp_audio_path}")
    # Clean up diar_output_dir and all its contents
    if os.path.exists(diar_output_dir):
        shutil.rmtree(diar_output_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {diar_output_dir}")
    # Clean up all chunk files in /tmp
    for f in os.listdir("/tmp"):
        if f.startswith("chunk_") and f.endswith(".wav"):
            try:
                os.remove(os.path.join("/tmp", f))
            except Exception:
                pass
    print("Processing complete!")
    return results


def main():
    """
    Main function to process audio files
    """
    # For local runs, use /tmp/output for compatibility with Lambda
    audio_folder = "/tmp/mp3"  # Place test files here for local test
    transcription_path = "/tmp/output/Transcription"
    diarization_path = "/tmp/output/Dializarition"
    os.makedirs(transcription_path, exist_ok=True)
    os.makedirs(diarization_path, exist_ok=True)
    mp3_files = glob.glob(os.path.join(audio_folder, "*.mp3"))
    print(f"Found {len(mp3_files)} audio files to process")
    if not mp3_files:
        print(f"No MP3 files found in {audio_folder}")
        print("Please check the path and ensure MP3 files are present.")
        return
    for i, mp3_file in enumerate(mp3_files):
        print(f"\nProcessing file {i+1}/{len(mp3_files)}: {os.path.basename(mp3_file)}")
        start_time = time.time()
        try:
            results = transcribe_and_diarize_mp3_two_speakers(
                mp3_file, diarization_path
            )
            if results:
                processing_time = time.time() - start_time
                print(f"Total processing time: {processing_time:.2f} seconds")
                duration_seconds = results.get("duration_seconds", 0)
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                duration_formatted = f"{minutes}:{seconds:02d}"
                audio_filename = os.path.basename(mp3_file)
                audio_filename_no_ext = os.path.splitext(audio_filename)[0]
                transcription_filename = f"{audio_filename_no_ext}.txt"
                transcription_filepath = os.path.join(
                    transcription_path, transcription_filename
                )
                with open(transcription_filepath, "w", encoding="utf-8") as f:
                    f.write(
                        f"File: {audio_filename} (Duration: {duration_formatted})\n\n"
                    )
                    f.write(results["full_transcription"])
                print(f"Transcription saved to: {transcription_filepath}")
            else:
                print("Processing failed.")
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")
            continue
    print("\nAll files processed successfully!")


if __name__ == "__main__":
    main()
