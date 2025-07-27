# cleanup_captions.ps1 - Clean up captions to only keep those with corresponding videos

Write-Host "Cleaning up captions..." -ForegroundColor Cyan

# Get all video files (without extension)
$videoFiles = Get-ChildItem "data/raw/videos/*.mp4" | ForEach-Object { $_.BaseName }

# Get all caption files
$captionFiles = Get-ChildItem "data/raw/captions/*.srt" | ForEach-Object { $_.BaseName }

# Find captions that don't have corresponding videos
$orphanedCaptions = @()
foreach ($caption in $captionFiles) {
    # Extract the video ID from caption filename (remove language suffix)
    $videoId = $caption -replace '\.(en|en-orig|en-en|en-es|en-cs|en-hu|en-hr)$', ''
    
    if ($videoId -notin $videoFiles) {
        $orphanedCaptions += $caption
    }
}

# Remove orphaned caption files
$removedCount = 0
foreach ($caption in $orphanedCaptions) {
    $captionPath = "data/raw/captions/$caption.srt"
    if (Test-Path $captionPath) {
        Remove-Item $captionPath -Force
        $removedCount++
        Write-Host "Removed: $caption.srt" -ForegroundColor Yellow
    }
}

Write-Host "`nCleanup complete!" -ForegroundColor Green
Write-Host "Removed $removedCount orphaned caption files" -ForegroundColor Green

# Show final counts
$finalVideoCount = (Get-ChildItem "data/raw/videos/*.mp4").Count
$finalCaptionCount = (Get-ChildItem "data/raw/captions/*.srt").Count

Write-Host "Final counts:" -ForegroundColor Cyan
Write-Host "- Videos: $finalVideoCount" -ForegroundColor White
Write-Host "- Captions: $finalCaptionCount" -ForegroundColor White 