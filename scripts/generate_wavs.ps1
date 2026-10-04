Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0

$outDir = Join-Path $PSScriptRoot "..\assets\audio"
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$pathSkipped = Join-Path $outDir "step_skipped.wav"
$synth.SetOutputToWaveFile($pathSkipped)
$synth.Speak("Alert. Step skipped. Please follow the correct protocol step.")

$pathWrong = Join-Path $outDir "wrong_sequence.wav"
$synth.SetOutputToWaveFile($pathWrong)
$synth.Speak("Protocol violation. Wrong sequence detected. Please check the current step suggestion.")

$pathDone = Join-Path $outDir "experiment_completed.wav"
$synth.SetOutputToWaveFile($pathDone)
$synth.Speak("Protocol verification complete. All milestones successfully achieved.")

$synth.Dispose()
Write-Host "All audio WAV files successfully generated in $outDir"
