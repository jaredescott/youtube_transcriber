# YouTube Song Transcriber for Genius.com

A Python tool to generate high-accuracy, line-by-line lyrics transcripts from YouTube songs, perfect for adding content to Genius.com.

## Features

- Downloads and transcribes YouTube videos to lyrics format
- Shows progress bars for download and transcription
- Intelligently formats lyrics with natural line breaks
- Supports multiple Whisper model sizes for different accuracy levels
- Optimized for music and lyrics transcription
- Parallel processing using multiple CPU cores
- GPU acceleration with CUDA support
- Fast mode for quicker transcription
- Beautifully animated progress bars
- Advanced logging with file output support

## Prerequisites

- Python 3.7+
- FFmpeg (required for audio processing)

### Installing FFmpeg

- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) or install with [Chocolatey](https://chocolatey.org/): `choco install ffmpeg`
- **macOS**: Install using [Homebrew](https://brew.sh/): `brew install ffmpeg`
- **Linux**: Install using package manager, e.g., `apt install ffmpeg` or `dnf install ffmpeg`

## Installation

1. Clone or download this repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Options

| Flag                    | Shortcut | Description                                                  | Default |
|-------------------------|----------|--------------------------------------------------------------|---------|
| `--output FILE`         | `-o`     | Save lyrics to this file                                     | stdout  |
| `--model SIZE`          | `-m`     | Choose model size (tiny, base, small, medium, large)         | medium  |
| `--keep-audio`          |          | Keep the downloaded audio file after processing              | False   |
| `--audio-output PATH`   |          | Specify path to save the downloaded audio                    |         |
| `--min-line-duration N` |          | Minimum pause duration (seconds) for line breaks             | 1.0     |
| `--raw-data`            |          | Save raw transcription data as JSON                          | False   |
| `--device TYPE`         |          | Device to use for inference (cpu or cuda)                    | auto    |
| `--chunk-size N`        |          | Process audio in chunks of this size in seconds              |         |
| `--fast`                |          | Enable fast mode (sacrifices some accuracy for speed)        | False   |
| `--max-duration N`      |          | Maximum duration (seconds) to transcribe from video start    |         |
| `--verbose`             | `-v`     | Enable verbose logging for debugging                         | False   |
| `--log-file FILE`       |          | Save logs to specified file in addition to console           |         |
| `--safe-mode`           |          | Use safe mode without word-level timestamps                  | False   |

### Examples

Transcribe song with higher accuracy for Genius.com:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -m large -o lyrics.txt
```

Customize line breaks with longer pauses:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --min-line-duration 2.0 -o lyrics.txt
```

Get raw data with timestamps:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --raw-data -o lyrics.txt
```

### Performance Optimization Examples

For fastest transcription with slightly lower quality:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --fast
```

Use GPU acceleration (if you have a CUDA-compatible GPU):
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device cuda
```

Process a long audio file in chunks:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --chunk-size 30
```

Only transcribe the first minute of a video:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max-duration 60
```

Fastest possible transcription (GPU + fast mode):
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --device cuda --fast
```

### Logging Options

Enable detailed debug logs:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -v
```

Save logs to a file for later analysis:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --log-file transcription.log
```

Both verbose console output and file logging:
```bash
python transcribe.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -v --log-file debug.log
```

### For Genius.com Users

This tool is optimized for creating lyrics for Genius.com:

1. Lines are intelligently separated based on natural pauses in the song
2. Default model is set to "medium" for better lyrics accuracy
3. Output is properly formatted and cleaned for easy copy/paste to Genius

## Performance Tips

- **GPU Acceleration**: If you have a NVIDIA GPU, use `--device cuda` for significantly faster transcription
- **Parallel Processing**: For long videos, use `--chunk-size 30` to process in 30-second chunks
- **Fast Mode**: The `--fast` flag will:
  - Use beam size 1 for faster inference
  - Automatically downgrade from medium to small model for better speed
  - Disable word-level timestamps for faster processing
- **Limit Duration**: For testing or when you only need part of a song, use `--max-duration 60` to only process the first minute
- **Model Selection**: For quick drafts, use `-m small` or `-m base`

## Virtual Environment Setup

For best results, install in a dedicated virtual environment:

```bash
# Create a virtual environment
python -m venv youtube_transcriber_env

# Activate it (Windows)
youtube_transcriber_env\Scripts\activate

# Activate it (Linux/Mac)
source youtube_transcriber_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, use uv for faster installation:

```bash
# Install uv
pip install uv

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Notes

- The script requires an internet connection to download the YouTube video and (first time only) to download the Whisper model.
- For song lyrics, the "medium" or "large" model provides much better accuracy and is highly recommended.
- Adjust the `--min-line-duration` parameter for different song styles:
  - Faster songs: try lower values (0.8-1.2 seconds)
  - Slower songs: try higher values (2.0-3.0 seconds)
- Progress bars show download and transcription progress in real-time
- Additional dependencies: For parallel processing, you need to install `librosa` (`pip install librosa`) 

## Sample Output

![Sample Lyrics Output](sample_output.png)

*Add an image of sample output here showing the formatted lyrics from a transcription.*

## Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) - The AI speech recognition model that powers the transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - For YouTube video downloading capabilities
- [PyTorch](https://pytorch.org/) - For the underlying machine learning framework
- [tqdm](https://github.com/tqdm/tqdm) - For the progress bar visualizations 