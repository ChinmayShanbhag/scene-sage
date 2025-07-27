# download_trailers.ps1  (ANSI/UTF-8 safe)

# 1 ── create target folders (mkdir -p equivalent)
New-Item -ItemType Directory -Force -Path "data/raw/videos"   | Out-Null
New-Item -ItemType Directory -Force -Path "data/raw/captions" | Out-Null

# 2 ── read non-empty YouTube IDs
$ids = Get-Content .\youtube_ids.txt | Where-Object { $_.Trim() }

# 3 ── choose yt-dlp command: EXE on PATH → use it, else fall back to python -m
$ytCmd = if (Get-Command yt-dlp -ErrorAction SilentlyContinue) {
             @('yt-dlp')
         } else {
             @('python','-m','yt_dlp')
         }

foreach ($id in $ids) {
    Write-Host "Downloading $id"

    & $ytCmd `
        "https://www.youtube.com/watch?v=$id" `
        -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best" `
        --merge-output-format mp4 `
        --write-sub --write-auto-sub --sub-lang "en.*" `
        --convert-subs srt `
        --sub-format srt `
        -o "data/raw/videos/${id}.%(ext)s"

    # move any English .srt subtitles to captions/
    Get-ChildItem "data/raw/videos/${id}.en*.srt" -ErrorAction SilentlyContinue |
        Move-Item -Destination "data/raw/captions" -Force -ErrorAction SilentlyContinue
}

Write-Host "`nFinished downloading." -ForegroundColor Green
