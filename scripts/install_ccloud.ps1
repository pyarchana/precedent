# Install the CockroachDB Cloud CLI on Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_ccloud.ps1
#
# Downloads the release zip, extracts ccloud.exe into %APPDATA%\ccloud, and
# reports the full path so it can be called without relying on PATH.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$version = "0.6.12"
$url = "https://binaries.cockroachdb.com/ccloud/ccloud_windows-amd64_$version.zip"
$dest = Join-Path $env:APPDATA "ccloud"
$zip = Join-Path $env:TEMP "ccloud.zip"
$extract = Join-Path $env:TEMP "ccloud_extract"

Write-Host "Downloading ccloud $version"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zip

Write-Host "Extracting"
if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
Expand-Archive -Path $zip -DestinationPath $extract -Force

# The archive layout has changed between releases, so find the binary rather
# than assuming where it sits.
$exe = Get-ChildItem -Path $extract -Filter ccloud.exe -Recurse | Select-Object -First 1
if (-not $exe) {
    throw "ccloud.exe was not found inside $zip"
}
Copy-Item -Force $exe.FullName -Destination $dest

$installed = Join-Path $dest "ccloud.exe"
Write-Host ""
Write-Host "Installed to $installed"
& $installed version
Write-Host ""
Write-Host "Authenticate with:"
Write-Host "    $installed auth login"
