# Scene Sage

Scene Sage is a video scene analysis and indexing system that processes video frames and creates searchable indices for scene content.

## Project Structure

```
scene-sage/
├── artifacts/           # Generated model artifacts and indices
│   ├── frame_meta.pkl  # Frame metadata
│   ├── frame_vectors.npy # Frame vector embeddings
│   └── scene_index.faiss # FAISS search index
├── data/
│   ├── frames/         # Extracted video frames
│   └── raw/           # Raw video data
├── cleanup_captions.ps1  # Script for cleaning caption data
└── download_trailers.ps1 # Script for downloading video content
```

## Features

- Frame extraction from videos
- Scene analysis and indexing
- FAISS-based similarity search
- Frame metadata management

## Getting Started

### Prerequisites

- Python 3.x
- PowerShell (for running utility scripts)
- Required Python packages (add requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/scene-sage.git
cd scene-sage
```

2. Install dependencies (once requirements.txt is added)
```bash
pip install -r requirements.txt
```

### Usage

1. Download video content:
```powershell
./download_trailers.ps1
```

2. Process and clean captions:
```powershell
./cleanup_captions.ps1
```

3. Run the scene analysis notebook:
```bash
jupyter notebook scene_sage_baseline.ipynb
```

## Data Management

- The `data/` directory contains video frames and raw video content
- The `artifacts/` directory contains generated model files and indices
- Both directories are git-ignored to prevent large binary files from being tracked
