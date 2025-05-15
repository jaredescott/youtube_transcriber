#!/usr/bin/env python3
import os
import argparse
import whisper
import tempfile
import yt_dlp
import json
import time
import multiprocessing
import torch
import logging
import sys
from tqdm import tqdm

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    # Need to use spawn when using CUDA to avoid "Cannot re-initialize CUDA in forked subprocess" error
    multiprocessing.set_start_method('spawn', force=True)

def setup_logger(verbose=False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    if verbose:
        # More detailed format for verbose mode
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        # Simpler format for normal mode
        formatter = logging.Formatter('%(message)s')
        
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def download_audio(url, output_path=None, max_duration=None):
    """Download audio from a YouTube video using yt-dlp."""
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading audio from: {url}")
    
    try:
        temp_dir = None
        
        if output_path is None:
            # Create a temporary file with .mp3 extension
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "audio.mp3")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128', # Use lower quality to speed up download
            }],
            'outtmpl': output_path.replace('.mp3', ''),
            'quiet': True,  # Be quiet so we can control output
            'no_warnings': True,  # No warnings to interfere with our progress bar
            'noprogress': True,  # We'll handle progress ourselves
        }
        
        # Add duration limit if specified
        if max_duration:
            ydl_opts['postprocessor_args'] = [
                '-ss', '0', '-t', str(max_duration)
            ]
        
        logger.info(f"Downloading audio to: {output_path}")
        
        # Create a progress bar that we'll manage
        pbar = None
        last_percent = -1
        filename = None
        
        def progress_hook(d):
            nonlocal pbar, last_percent, filename
            
            try:
                if d['status'] == 'downloading':
                    # Get filename if we don't have it yet
                    if filename is None and 'filename' in d:
                        filename = os.path.basename(d['filename'])
                    
                    # Calculate percentage
                    if 'total_bytes' in d and d['total_bytes'] > 0:
                        percent = int(d['downloaded_bytes'] / d['total_bytes'] * 100)
                        
                        # Create progress bar if it doesn't exist
                        if pbar is None:
                            # Create a simpler TQDM progress bar with fewer formatting issues
                            pbar = tqdm(
                                total=100, 
                                desc=f"Downloading audio",
                                unit="%"
                            )
                        
                        # Update if percentage changed
                        if percent > last_percent:
                            # Update bar by the difference
                            pbar.update(percent - last_percent)
                            last_percent = percent
                    
                    # Handle case where we don't have total bytes
                    elif '_percent_str' in d and pbar is None:
                        # Try to extract percentage from string
                        try:
                            percent_str = d['_percent_str'].strip()
                            if percent_str.endswith('%'):
                                percent = float(percent_str[:-1])
                                # Create indeterminate progress bar
                                pbar = tqdm(
                                    total=100,
                                    desc=f"Downloading audio",
                                    unit="%"
                                )
                        except (ValueError, AttributeError):
                            pass
                
                elif d['status'] == 'finished':
                    # Complete and close the progress bar
                    if pbar:
                        # Make sure we're at 100%
                        if last_percent < 100:
                            pbar.update(100 - last_percent)
                        pbar.close()
                    logger.info("Converting to mp3...")
            except Exception as e:
                # Log the error but don't let it crash the download
                logger.debug(f"Progress bar error: {str(e)}")
                # If there's an issue with the progress bar, close it and continue
                if pbar:
                    try:
                        pbar.close()
                    except:
                        pass
                    pbar = None
        
        # Set the progress hook
        ydl_opts['progress_hooks'] = [progress_hook]
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', None)
        
        # The actual output path with extension added by yt-dlp
        actual_output_path = output_path.replace('.mp3', '') + '.mp3'
        
        # Verify the file exists and log its size
        if os.path.exists(actual_output_path):
            file_size_mb = os.path.getsize(actual_output_path) / (1024 * 1024)
            logger.info(f"Downloaded MP3: {actual_output_path} ({file_size_mb:.2f}MB)")
        else:
            logger.warning(f"Expected output file not found: {actual_output_path}")
        
        return actual_output_path, video_title
    
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        return None, None

def transcribe_segment(segment_data):
    """Transcribe a single audio segment. This function ensures each process loads its own model."""
    segment, model_size, transcribe_kwargs = segment_data
    
    # Load a new model instance within each process to avoid CUDA sharing issues
    model = whisper.load_model(model_size, device="cpu")  # Force CPU for multiprocessing
    
    # Process this segment
    result = model.transcribe(segment, **transcribe_kwargs)
    return result

def transcribe_audio(audio_path, model_size="base", device=None, chunk_size=None, fast_mode=False):
    """Transcribe audio using Whisper model with line breaks for lyrics."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Determine device to use
    original_device_request = device  # Store the original request
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # For large models, check the GPU memory
    is_large_model = model_size in ["medium", "large"]
    if device == "cuda" and is_large_model:
        try:
            # Check available GPU memory
            free_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            model_size_mb = {"medium": 2500, "large": 5000}.get(model_size, 1500)
            
            if free_mem_mb < model_size_mb * 1.5:  # Need 1.5x the model size free
                logger.warning(f"Limited GPU memory detected ({free_mem_mb:.0f}MB). {model_size} model needs ~{model_size_mb}MB.")
                logger.warning(f"Consider using a smaller model or --device cpu to avoid out-of-memory errors.")
        except Exception:
            # If we can't check memory, just proceed
            pass
    
    # Get estimated loading time for the model (very rough estimate)
    loading_times = {
        "tiny": 1, 
        "base": 2, 
        "small": 8, 
        "medium": 15, 
        "large": 25
    }
    # Adjust for CPU vs GPU (CPU is ~2x slower)
    if device == "cpu":
        for key in loading_times:
            loading_times[key] *= 2
    
    # Get estimated load time for this model
    est_load_time = loading_times.get(model_size, 5)
    print(f"\n⏳ Loading {model_size} model on {device}... (may take ~{est_load_time}s)")
    
    # We'll track loading time for better future estimates
    model_load_start = time.time()
    
    try:
        # Create a simple loading spinner with ASCII animation
        from itertools import cycle
        import threading
        import sys
        
        # Set up a spinner animation
        spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_stop = False
        
        def show_spinner():
            spinner = cycle(spinner_frames)
            elapsed = 0
            sys.stdout.write('\r')
            while not spinner_stop:
                elapsed = int(time.time() - model_load_start)
                frame = next(spinner)
                sys.stdout.write(f"\r{frame} Loading model: {elapsed}s elapsed")
                sys.stdout.flush()
                time.sleep(0.1)
            # Clear the line when done
            sys.stdout.write('\r' + ' ' * 50 + '\r')
            sys.stdout.flush()
        
        # Start the spinner in a separate thread
        spinner_thread = threading.Thread(target=show_spinner)
        spinner_thread.daemon = True
        spinner_thread.start()
        
        # Actually load the model
        try:
            model = whisper.load_model(model_size, device=device)
        except RuntimeError as e:
            # If CUDA out of memory, fall back to CPU
            if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                spinner_stop = True  # Stop the spinner thread
                spinner_thread.join(timeout=1.0)  # Wait for it to finish
                
                logger.warning(f"GPU out of memory error: {e}")
                logger.warning("Falling back to CPU. This will be slower but more reliable.")
                device = "cpu"
                
                # Restart spinner for CPU loading
                spinner_stop = False
                model_load_start = time.time()
                spinner_thread = threading.Thread(target=show_spinner)
                spinner_thread.daemon = True
                spinner_thread.start()
                
                # Try loading on CPU
                model = whisper.load_model(model_size, device=device)
            else:
                # Re-raise if it's not a memory error
                spinner_stop = True
                spinner_thread.join(timeout=1.0)
                raise
        
        # Stop the spinner
        spinner_stop = True
        spinner_thread.join(timeout=1.0)
        
        # Report loading time
        model_load_time = time.time() - model_load_start
        logger.info(f"Model loaded in {model_load_time:.2f}s")
        
        # Set computation type to float16 if using CUDA for better performance
        compute_type = "float16" if device == "cuda" else "float32"
        
        # Default transcription options - use simpler options to avoid errors
        transcribe_kwargs = {
            'fp16': compute_type == "float16",
            # Word timestamps often cause errors, so disable them by default
            'word_timestamps': False
        }
        
        # If fast mode is enabled, use even more minimal settings
        if fast_mode:
            transcribe_kwargs.update({
                'beam_size': 1,
                'best_of': 1
            })
        
        # Check if chunking is enabled
        if chunk_size:
            logger.info(f"Processing audio in chunks of {chunk_size} seconds...")
            
            # Load full audio
            import librosa
            logger.info("Loading audio file...")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio) / sr
            logger.info(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Calculate chunk size in samples
            chunk_samples = int(chunk_size * sr)
            
            # Split audio into chunks
            chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
            total_chunks = len(chunks)
            logger.info(f"Processing {total_chunks} chunks...")
            
            # Process chunks sequentially
            results = []
            # Create a progress bar for chunk processing
            with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as chunk_progress:
                for i, chunk in enumerate(chunks):
                    current_chunk = i + 1
                    progress_percent = int(100 * current_chunk / total_chunks)
                    logger.debug(f"Processing chunk {current_chunk}/{total_chunks} ({progress_percent}%)")
                    
                    # Process this chunk
                    try:
                        result = model.transcribe(chunk, **transcribe_kwargs)
                    except RuntimeError as e:
                        # If CUDA out of memory during processing, try that chunk on CPU
                        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                            logger.warning(f"GPU out of memory on chunk {current_chunk}. Processing this chunk on CPU.")
                            
                            # Create a CPU model for this chunk if needed
                            if device != "cpu":
                                # Use CPU just for this chunk
                                cpu_model = whisper.load_model(model_size, device="cpu")
                                result = cpu_model.transcribe(chunk, **transcribe_kwargs)
                                # Clean up - remove references to allow garbage collection
                                del cpu_model
                                import gc
                                gc.collect()
                                # If using CUDA, try to free GPU memory
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            else:
                                # If we're already on CPU, just re-raise the error
                                raise
                        else:
                            # Re-raise if it's not a memory error
                            raise

                    results.append(result)
                    logger.debug(f"Chunk {current_chunk}/{total_chunks} completed")
                    # Update progress bar
                    chunk_progress.update(1)
                    
            # Merge results
            logger.info("Merging chunks...")
            final_result = {
                "text": " ".join(r["text"] for r in results),
                "segments": []
            }
            
            # Adjust timestamps and concatenate segments
            time_offset = 0
            for chunk_idx, chunk_result in enumerate(results):
                for segment in chunk_result.get("segments", []):
                    # Adjust start and end times
                    segment["start"] += time_offset
                    segment["end"] += time_offset
                    final_result["segments"].append(segment)
                
                # Update time offset for next chunk
                if chunk_result.get("segments"):
                    time_offset = final_result["segments"][-1]["end"]
            
            result = final_result
        else:
            # Standard single-process transcription with the specified device
            logger.info("Starting transcription...")
            
            # Similar to model loading, create a spinner for transcription
            transcribe_start = time.time()
            spinner_stop = False
            
            def show_transcribe_spinner():
                spinner = cycle(spinner_frames)
                elapsed = 0
                sys.stdout.write('\r')
                while not spinner_stop:
                    elapsed = int(time.time() - transcribe_start)
                    frame = next(spinner)
                    sys.stdout.write(f"\r{frame} Transcribing: {elapsed}s elapsed")
                    sys.stdout.flush()
                    time.sleep(0.1)
                # Clear the line when done
                sys.stdout.write('\r' + ' ' * 50 + '\r')
                sys.stdout.flush()
            
            # Start the spinner in a separate thread
            spinner_thread = threading.Thread(target=show_transcribe_spinner)
            spinner_thread.daemon = True
            spinner_thread.start()
            
            # Actually do the transcription
            try:
                result = model.transcribe(audio_path, **transcribe_kwargs)
            except RuntimeError as e:
                # If CUDA out of memory, fall back to CPU
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    spinner_stop = True  # Stop the spinner thread
                    spinner_thread.join(timeout=1.0)
                    
                    logger.warning(f"GPU out of memory during transcription: {e}")
                    logger.warning("Switching to CPU. This will be slower but more reliable.")
                    
                    # Free GPU memory if possible
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Start new spinner for CPU transcription
                    spinner_stop = False
                    transcribe_start = time.time()
                    spinner_thread = threading.Thread(target=show_transcribe_spinner)
                    spinner_thread.daemon = True
                    spinner_thread.start()
                    
                    # Try with CPU
                    if device != "cpu":
                        device = "cpu"
                        model = whisper.load_model(model_size, device="cpu")
                        result = model.transcribe(audio_path, **transcribe_kwargs)
                    else:
                        # If we're already on CPU, just re-raise the error
                        spinner_stop = True
                        spinner_thread.join(timeout=1.0)
                        raise
                else:
                    # Re-raise if it's not a memory error
                    spinner_stop = True
                    spinner_thread.join(timeout=1.0)
                    raise
            
            # Stop the spinner
            spinner_stop = True
            spinner_thread.join(timeout=1.0)
            
            # Report transcription time
            transcribe_time = time.time() - transcribe_start
            logger.info(f"File transcribed in {transcribe_time:.2f}s")
            
            # Add debug logging to track result
            if result is None:
                logger.error("CRITICAL: Transcription result is None")
            else:
                logger.debug(f"Transcription successful, result type: {type(result)}")
                if isinstance(result, dict):
                    logger.debug(f"Result contains keys: {list(result.keys())}")
        
        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
        
        # Log the device actually used (might be different from requested)
        if device != original_device_request and original_device_request is not None:
            logger.info(f"Note: Transcription was performed on {device} instead of the requested {original_device_request}")
        
        # Final check before returning
        if 'result' not in locals():
            logger.error("CRITICAL ERROR: 'result' variable is not defined before return")
            return {"text": "Transcription failed - result variable was not defined"}
            
        logger.debug("Returning transcription result from function")
        return result
    except Exception as e:
        # If we have an active spinner, stop it
        if 'spinner_stop' in locals() and spinner_thread.is_alive():
            spinner_stop = True
            spinner_thread.join(timeout=1.0)
            
        logger.error(f"Error during transcription: {e}")
        end_time = time.time()
        logger.debug(f"Transcription attempt took {end_time - start_time:.2f} seconds before failing")
        
        # Try basic CPU transcription as absolute last resort
        try:
            logger.info("Attempting basic CPU transcription as last resort...")
            start_time = time.time()
            model = whisper.load_model(model_size, device="cpu")
            # Use minimal options
            basic_kwargs = {
                'fp16': False,
                'word_timestamps': False
            }
            result = model.transcribe(audio_path, **basic_kwargs)
            end_time = time.time()
            logger.info(f"Fallback transcription completed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Even fallback transcription failed: {e}")
            # Return at least the text if possible
            return {"text": "Transcription failed due to technical errors."}

def format_lyrics(result, min_line_duration=0.5):
    """Format transcription into lyrics with line breaks based on pauses."""
    logger = logging.getLogger(__name__)
    
    # Check if result is None and handle gracefully
    if result is None:
        logger.error("Transcription result is None. Unable to format lyrics.")
        return ["[No transcription available]"]
    
    if isinstance(result, str):
        # If result is just the transcript text, return as is
        return [result]
    
    # Additional debug logging to help identify issues
    logger.debug(f"Processing transcription result of type: {type(result)}")
    if isinstance(result, dict):
        logger.debug(f"Result contains keys: {list(result.keys())}")
    
    formatted_lyrics = []
    current_line = []
    last_end_time = 0
    
    # Safely check if we have segments with timestamps
    if isinstance(result, dict) and "segments" in result and result["segments"]:
        # First try to use word-level timestamps if available
        has_words = False
        for segment in result["segments"]:
            if "words" in segment and segment["words"]:
                has_words = True
                break
        
        if has_words:
            logger.debug("Using word-level timestamps for line breaks")
            # Use word-level timestamps
            for segment in result["segments"]:
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        # If there's a significant pause, start a new line
                        if word["start"] - last_end_time > min_line_duration and current_line:
                            formatted_lyrics.append(" ".join(current_line))
                            current_line = []
                        
                        current_line.append(word["word"].strip())
                        last_end_time = word["end"]
                else:
                    # No words in this segment, add as a separate line
                    if segment["text"].strip():
                        formatted_lyrics.append(segment["text"].strip())
        else:
            logger.debug("Using segment-level timestamps for line breaks")
            # Fall back to segment-level timestamps
            last_segment_end = 0
            for segment in result["segments"]:
                text = segment["text"].strip()
                if not text:
                    continue
                
                # If there's a significant pause between segments, start a new line
                if segment["start"] - last_segment_end > min_line_duration and formatted_lyrics:
                    formatted_lyrics.append(text)
                elif formatted_lyrics:
                    # Append to the previous line
                    formatted_lyrics[-1] += " " + text
                else:
                    # First line
                    formatted_lyrics.append(text)
                
                last_segment_end = segment["end"]
    elif isinstance(result, dict) and "text" in result:
        logger.debug("No segments available, using punctuation for line breaks")
        # Fallback to just the text and split on punctuation
        text = result["text"]
        # Split on periods, question marks, exclamation marks, and newlines
        import re
        lines = re.split(r'(?<=[.!?])\s+', text)
        formatted_lyrics = [line.strip() for line in lines if line.strip()]
    else:
        logger.warning("Unexpected result format, returning raw text if possible")
        # Try to extract any text we can find
        if isinstance(result, dict):
            text = result.get("text", str(result))
        else:
            text = str(result)
        return [text]
    
    # Add the last line if there's content
    if current_line:
        formatted_lyrics.append(" ".join(current_line))
    
    # Clean up the lyrics
    cleaned_lyrics = []
    for line in formatted_lyrics:
        line = line.strip()
        if line and not line.isspace():
            # Remove redundant spaces and clean up punctuation
            cleaned_line = ' '.join(line.split())
            cleaned_lyrics.append(cleaned_line)
    
    if not cleaned_lyrics:
        logger.warning("No lyrics could be extracted from the transcription")
        return ["[No transcription text available]"]
    
    logger.debug(f"Formatted {len(cleaned_lyrics)} lines of lyrics")
    return cleaned_lyrics

def save_raw_json(result, output_path):
    """Save the raw transcription data to a JSON file."""
    logger = logging.getLogger(__name__)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Raw transcription data saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos into lyrics format")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", help="Output file for transcript (default: stdout)")
    parser.add_argument("--model", "-m", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: medium)")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the downloaded audio file")
    parser.add_argument("--audio-output", help="Path to save the audio file")
    parser.add_argument("--raw-data", action="store_true", help="Save raw transcription data as JSON")
    parser.add_argument("--min-line-duration", type=float, default=1.0, 
                        help="Minimum pause duration (seconds) to start a new line (default: 1.0)")
    parser.add_argument("--safe-mode", action="store_true", 
                        help="Use safe mode without word-level timestamps (more compatible)")
    parser.add_argument("--device", choices=["cpu", "cuda"], 
                        help="Device to use for inference (default: auto-detect)")
    parser.add_argument("--chunk-size", type=float,
                        help="Process audio in chunks of this size in seconds")
    parser.add_argument("--fast", action="store_true",
                        help="Enable fast mode (sacrifices some accuracy for speed)")
    parser.add_argument("--max-duration", type=float, 
                        help="Maximum duration (in seconds) to transcribe from the start of the video")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging for debugging")
    parser.add_argument("--log-file", 
                        help="Path to save log file in addition to console output")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(args.verbose)
    
    # Add file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    # Download audio
    if args.audio_output:
        audio_path, video_title = download_audio(args.url, args.audio_output, args.max_duration)
    else:
        audio_path, video_title = download_audio(args.url, None, args.max_duration)
    
    if not audio_path:
        logger.error("Failed to download audio. Exiting.")
        return
    
    # Choose a smaller model if fast mode is enabled
    if args.fast and args.model == "medium":
        logger.info("Fast mode enabled: Using 'small' model instead of 'medium' for better speed")
        model_size = "small"
    else:
        model_size = args.model
    
    # If CUDA available and device not explicitly specified, recommend GPU
    if args.device is None and torch.cuda.is_available():
        logger.info("CUDA GPU detected! For faster transcription, consider using --device cuda")
    
    # Transcribe audio
    result = transcribe_audio(
        audio_path, 
        model_size=model_size,
        device=args.device,
        chunk_size=args.chunk_size,
        fast_mode=args.fast
    )
    
    # Format the lyrics
    lyrics_lines = format_lyrics(result, args.min_line_duration)
    lyrics_text = '\n'.join(lyrics_lines)
    
    # Save raw transcription data if requested
    if args.raw_data and args.output:
        raw_output = args.output + ".json"
        save_raw_json(result, raw_output)
    
    # Output formatted lyrics
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"Lyrics for: {video_title}\n\n")
            f.write(lyrics_text)
        logger.info(f"Lyrics saved to: {args.output}")
    else:
        logger.info(f"\nLyrics for: {video_title}\n")
        print(lyrics_text)
    
    # Clean up temporary audio file if not keeping it
    if not args.keep_audio and not args.audio_output and os.path.exists(audio_path):
        os.remove(audio_path)
        logger.debug(f"Temporary audio file deleted: {audio_path}")
        
        # Clean up any temporary directory created
        parent_dir = os.path.dirname(audio_path)
        if parent_dir.startswith(tempfile.gettempdir()) and os.path.exists(parent_dir):
            import shutil
            shutil.rmtree(parent_dir)
            logger.debug(f"Temporary directory cleaned up: {parent_dir}")

if __name__ == "__main__":
    main() 