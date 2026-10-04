$s = New-Object -ComObject SAPI.SpVoice
$fs = New-Object -ComObject SAPI.SpFileStream
$fs.Open('assets/audio/next_step.wav', 3)
$s.AudioOutputStream = $fs
$s.Speak('Next protocol step ready.')
$fs.Close()
Write-Host "Audio written successfully."
