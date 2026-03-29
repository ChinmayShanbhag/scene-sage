"""
SceneSage Demo Backend
Lazy-loaded FastAPI server — models and FAISS index are only loaded
when the first /api/search request arrives.
"""

from __future__ import annotations

import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import open_clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FRAMES_DIR = Path(__file__).resolve().parent.parent / "data" / "frames"
WEB_DIR = Path(__file__).resolve().parent

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".avi", ".mov"}

K_SEARCH = 50
K_TRAILERS = 5

app = FastAPI(title="SceneSage Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _Models:
    """Singleton that holds models + index. Loaded once on first request."""

    def __init__(self) -> None:
        self.loaded = False
        self.clip_model = None
        self.preprocess = None
        self.txt_model = None
        self.index = None
        self.meta: list[tuple[str, str]] = []
        self.device = "cpu"

    def ensure_loaded(self) -> None:
        if self.loaded:
            return

        required_files = [
            ARTIFACTS_DIR / "scene_index.faiss",
            ARTIFACTS_DIR / "frame_meta.pkl",
        ]
        missing = [str(f) for f in required_files if not f.exists()]
        if missing:
            raise HTTPException(
                503,
                "Model artifacts not found. Contact the developer to let them know."
                f"artifacts/. Missing: {', '.join(f.name for f in required_files if not f.exists())}",
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SceneSage] Loading models on {self.device} …")

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai",
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        if self.device == "cuda":
            self.clip_model = self.clip_model.half()

        self.txt_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        print("[SceneSage] Loading FAISS index …")
        self.index = faiss.read_index(str(ARTIFACTS_DIR / "scene_index.faiss"))

        with open(ARTIFACTS_DIR / "frame_meta.pkl", "rb") as f:
            self.meta = pickle.load(f)

        self.loaded = True
        print(f"[SceneSage] Ready — {self.index.ntotal:,} vectors, "
              f"{len(set(v for v, _ in self.meta))} trailers")


models = _Models()


def _extract_frames_from_video(video_path: Path, fps: int = 1) -> Path:
    tmp = Path(tempfile.mkdtemp())
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(tmp / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True, timeout=60)
    return tmp


def _extract_single_frame_from_video(video_path: Path) -> Path:
    """Extract a single representative frame from the middle of a video."""
    tmp = Path(tempfile.mkdtemp())
    out_path = tmp / "frame.jpg"
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", "select=eq(n\\,0)",
        "-frames:v", "1",
        "-ss", "5",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=False, timeout=30)
    if not out_path.exists():
        cmd_fallback = [
            "ffmpeg", "-v", "error",
            "-i", str(video_path),
            "-frames:v", "1",
            str(out_path),
        ]
        subprocess.run(cmd_fallback, check=True, timeout=30)
    return tmp


@torch.no_grad()
def _encode_images(image_paths: list[Path]) -> np.ndarray:
    """Encode images → (N, 896) float32 vectors (512 CLIP + 384 zeros)."""
    m = models
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensor = m.preprocess(img)
        if m.device == "cuda":
            tensor = tensor.half()
        imgs.append(tensor)

    batch = torch.stack(imgs).to(m.device, non_blocking=True)
    v_img = m.clip_model.encode_image(batch).cpu().float().numpy()

    zero_txt = np.zeros((v_img.shape[0], 384), dtype="float32")
    v_full = np.hstack([v_img, zero_txt]).astype("float32")
    v_full = np.ascontiguousarray(v_full)
    faiss.normalize_L2(v_full)
    return v_full


def _search(q_vecs: np.ndarray, top_n: int = K_TRAILERS) -> list[dict]:
    """Run FAISS search and deduplicate at trailer level."""
    m = models
    D, I = m.index.search(q_vecs, K_SEARCH)

    from collections import Counter
    votes: Counter[str] = Counter()
    scores: dict[str, list[float]] = {}
    best_frame: dict[str, tuple[str, float]] = {}

    for row in range(len(q_vecs)):
        seen: set[str] = set()
        for col_idx in range(K_SEARCH):
            flat_idx = int(I[row][col_idx])
            if flat_idx < 0 or flat_idx >= len(m.meta):
                continue
            vid, fname = m.meta[flat_idx]
            if vid not in seen:
                score = float(D[row][col_idx])
                votes[vid] += 1
                scores.setdefault(vid, []).append(score)
                if vid not in best_frame or score > best_frame[vid][1]:
                    best_frame[vid] = (fname, score)
                seen.add(vid)
                if len(seen) >= 5:
                    break

    ranked = [
        (vid, votes[vid], float(np.mean(scores[vid])))
        for vid in votes
    ]
    ranked.sort(key=lambda x: (-x[1], -x[2]))

    results = []
    for vid, vote_count, mean_cos in ranked[:top_n]:
        frame_name, top_score = best_frame[vid]
        frame_path = FRAMES_DIR / vid / frame_name
        results.append({
            "video_id": vid,
            "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            "votes": vote_count,
            "mean_cosine": round(mean_cos, 4),
            "top_cosine": round(top_score, 4),
            "best_frame": frame_name,
            "has_local_frame": frame_path.exists(),
        })

    return results


def _download_youtube_thumbnail(url: str) -> Path:
    """Download a single frame from a YouTube video via yt-dlp."""
    import re
    match = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)"
        r"([A-Za-z0-9_-]{11})",
        url,
    )
    if not match:
        raise HTTPException(400, "Invalid YouTube URL")

    video_id = match.group(1)
    tmp = Path(tempfile.mkdtemp())
    out_path = tmp / f"{video_id}.mp4"

    try:
        subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "-f", "worst[ext=mp4]",
                "--download-sections", "*0-10",
                "-o", str(out_path),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            check=True,
            timeout=120,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(
            502,
            f"Failed to download YouTube video: {exc}",
        ) from exc

    return tmp


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": models.loaded,
        "index_vectors": models.index.ntotal if models.loaded else 0,
    }


@app.post("/api/search")
async def search(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
):
    if file is None and not youtube_url:
        raise HTTPException(400, "Provide an image/video file or a YouTube URL")

    models.ensure_loaded()
    tmp_dir: Optional[Path] = None

    try:
        if youtube_url:
            tmp_dir = _download_youtube_thumbnail(youtube_url)
            video_files = list(tmp_dir.glob("*.mp4"))
            if not video_files:
                raise HTTPException(502, "No video downloaded from YouTube")
            frame_dir = _extract_single_frame_from_video(video_files[0])
            image_paths = sorted(frame_dir.glob("*.jpg"))
            if not image_paths:
                raise HTTPException(502, "Could not extract frames from YouTube video")

        elif file is not None:
            ext = Path(file.filename or "").suffix.lower()
            is_image = ext in ALLOWED_IMAGE_EXT
            is_video = ext in ALLOWED_VIDEO_EXT

            if not is_image and not is_video:
                raise HTTPException(
                    400,
                    f"Unsupported file type '{ext}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT))}",
                )

            content = await file.read()
            max_bytes = MAX_IMAGE_BYTES if is_image else MAX_VIDEO_BYTES
            if len(content) > max_bytes:
                limit_mb = max_bytes / (1024 * 1024)
                raise HTTPException(
                    413,
                    f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
                    f"Max for {'images' if is_image else 'videos'}: {limit_mb:.0f} MB",
                )

            tmp_dir = Path(tempfile.mkdtemp())
            saved_path = tmp_dir / f"upload{ext}"
            saved_path.write_bytes(content)

            if is_image:
                image_paths = [saved_path]
            else:
                frame_dir = _extract_frames_from_video(saved_path, fps=1)
                image_paths = sorted(frame_dir.glob("*.jpg"))
                if not image_paths:
                    raise HTTPException(422, "Could not extract any frames from video")
                if len(image_paths) > 30:
                    step = len(image_paths) // 30
                    image_paths = image_paths[::step][:30]

        q_vecs = _encode_images(image_paths)
        results = _search(q_vecs)

        return {
            "results": results,
            "query_frames": len(image_paths),
            "index_size": models.index.ntotal,
            "trailer_count": len(set(v for v, _ in models.meta)),
        }

    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/{path:path}")
async def serve_static(request: Request, path: str):
    """Serve static files from the web directory, falling back to index.html."""
    if path == "" or path == "/":
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    file_path = WEB_DIR / path
    if file_path.is_file() and WEB_DIR in file_path.resolve().parents:
        media = None
        suffix = file_path.suffix.lower()
        media_types = {
            ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
            ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
            ".svg": "image/svg+xml", ".ico": "image/x-icon", ".woff2": "font/woff2",
        }
        media = media_types.get(suffix)
        return FileResponse(file_path, media_type=media)

    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
