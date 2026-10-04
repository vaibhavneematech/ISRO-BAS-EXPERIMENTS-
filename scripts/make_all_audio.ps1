$voice = New-Object -ComObject SAPI.SpVoice
$dir = "assets\audio"

$clips = @{
    "step_complete.wav"       = "Step completed successfully."
    "wrong_sequence.wav"      = "Warning. Wrong step sequence detected."
    "step_skipped.wav"        = "Caution. Required protocol step was skipped."
    "next_step.wav"           = "Please proceed to the next protocol directive."
    "experiment_complete.wav" = "Protocol sequence completed. Experiment validated."
}

foreach ($name in $clips.Keys) {
    $outPath = Join-Path $dir $name
    $stream = New-Object -ComObject SAPI.SpFileStream
    # 3 = SSFMCreateForWrite
    $stream.Open($outPath, 3)
    $voice.AudioOutputStream = $stream
    $voice.Speak($clips[$name])
    $stream.Close()
    Write-Host "Generated $outPath"
}
