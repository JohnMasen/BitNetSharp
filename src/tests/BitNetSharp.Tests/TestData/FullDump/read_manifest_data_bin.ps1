<#
.SYNOPSIS
Reads entries from a manifest.json + data.bin pair produced by bitnet_cached_attention_dump.

.DESCRIPTION
This script reads the manifest, locates data.bin, and either lists all entries or decodes one entry
using the entry's offset, byte_length, and dtype.

Supported dtypes:
- F32
- F16
- inline_json

.PARAMETER ManifestPath
Path to manifest.json.

.PARAMETER EntryName
Entry Name to decode from data.bin. If omitted, the script lists entries.

.PARAMETER EmbeddingOutput
Shortcut for EntryName = embedding_output_first_32.

.PARAMETER FirstLayerAttentionNormOutput
Shortcut for EntryName = first_layer_attention_norm_output_first_32.

.PARAMETER FirstLayerFeedforwardNormOutput
Shortcut for EntryName = first_layer_feedforward_norm_output_first_32.

.PARAMETER FirstLayerFeedforwardInput
Shortcut for EntryName = first_layer_feedforward_input_first_32.

.PARAMETER FirstLayerFeedforwardOutput
Shortcut for EntryName = first_layer_feedforward_output_first_32.

.PARAMETER FirstLayerFeedforwardSubNormOutput
Shortcut for EntryName = first_layer_feedforward_sub_norm_output_first_32.

.PARAMETER FirstLayerFeedforwardDownOutput
Shortcut for EntryName = first_layer_feedforward_down_output_first_32.

.PARAMETER FirstLayerFeedforwardOutputRuntimeSemantic
Shortcut for EntryName = first_layer_feedforward_output_runtime_semantic_first_32.

.PARAMETER FinalNormOutput
Shortcut for EntryName = final_norm_output_first_32.

.PARAMETER LmHeadOutputLogits
Shortcut for EntryName = lm_head_output_logits_first_32.

.PARAMETER FirstLayerAttentionOutputFullFirst64
Shortcut for EntryName = first_layer_attention_output_full_first_64.

.PARAMETER FirstLayerFeedforwardNormOutputFullFirst64
Shortcut for EntryName = first_layer_feedforward_norm_output_full_first_64.

.PARAMETER FirstLayerFeedforwardInputFullFirst64
Shortcut for EntryName = first_layer_feedforward_input_full_first_64.

.PARAMETER FirstLayerFeedforwardOutputFullFirst64
Shortcut for EntryName = first_layer_feedforward_output_full_first_64.

.PARAMETER FirstLayerFeedforwardSubNormOutputFullFirst64
Shortcut for EntryName = first_layer_feedforward_sub_norm_output_full_first_64.

.PARAMETER FirstLayerFeedforwardDownOutputFullFirst64
Shortcut for EntryName = first_layer_feedforward_down_output_full_first_64.

.PARAMETER FirstLayerFeedforwardOutputRuntimeSemanticFullFirst64
Shortcut for EntryName = first_layer_feedforward_output_runtime_semantic_full_first_64.

.PARAMETER FirstLayerFeedforwardOutputContext
Shortcut for EntryName = first_layer_feedforward_output_context.

.PARAMETER FirstLayerFeedforwardInputContext
Shortcut for EntryName = first_layer_feedforward_input_context.

.PARAMETER FirstLayerFeedforwardSubNormOutputContext
Shortcut for EntryName = first_layer_feedforward_sub_norm_output_context.

.PARAMETER FirstLayerFeedforwardDownOutputContext
Shortcut for EntryName = first_layer_feedforward_down_output_context.

.PARAMETER FirstLayerFeedforwardOutputRuntimeSemanticContext
Shortcut for EntryName = first_layer_feedforward_output_runtime_semantic_context.

.PARAMETER ListOnly
List entries without decoding tensor payloads.

.PARAMETER AsJson
When decoding one entry, emit the result as JSON.

.PARAMETER Help
Show this help text.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -ListOnly

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -EntryName first_layer_qcur_before_rope_first_32 -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -EmbeddingOutput -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -LmHeadOutputLogits -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerAttentionOutputFullFirst64 -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardOutputFullFirst64 -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardInput -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardDownOutputFullFirst64 -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardInputContext -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardOutputRuntimeSemantic -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardOutputRuntimeSemanticFullFirst64 -AsJson

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\read_manifest_data_bin.ps1 -ManifestPath .\manifest.json -FirstLayerFeedforwardOutputRuntimeSemanticContext -AsJson

.NOTES
If generated_files.data.bin.path is present in the manifest, that path is used first.
Otherwise the script falls back to a sibling data.bin next to manifest.json.
#>
param(
    [string]$ManifestPath,

    [string]$EntryName,

    [switch]$EmbeddingOutput,

    [switch]$FirstLayerAttentionNormOutput,

    [switch]$FirstLayerFeedforwardInput,

    [switch]$FirstLayerFeedforwardNormOutput,

    [switch]$FirstLayerFeedforwardOutput,

    [switch]$FirstLayerFeedforwardSubNormOutput,

    [switch]$FirstLayerFeedforwardDownOutput,

    [switch]$FirstLayerFeedforwardOutputRuntimeSemantic,

    [switch]$FinalNormOutput,

    [switch]$LmHeadOutputLogits,

    [switch]$FirstLayerAttentionOutputFullFirst64,

    [switch]$FirstLayerFeedforwardInputFullFirst64,

    [switch]$FirstLayerFeedforwardNormOutputFullFirst64,

    [switch]$FirstLayerFeedforwardOutputFullFirst64,

    [switch]$FirstLayerFeedforwardSubNormOutputFullFirst64,

    [switch]$FirstLayerFeedforwardDownOutputFullFirst64,

    [switch]$FirstLayerFeedforwardOutputRuntimeSemanticFullFirst64,

    [switch]$FirstLayerFeedforwardInputContext,

    [switch]$FirstLayerFeedforwardOutputContext,

    [switch]$FirstLayerFeedforwardSubNormOutputContext,

    [switch]$FirstLayerFeedforwardDownOutputContext,

    [switch]$FirstLayerFeedforwardOutputRuntimeSemanticContext,

    [switch]$ListOnly,

    [switch]$AsJson,

    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Help) {
    Get-Help -Full $MyInvocation.MyCommand.Path
    return
}

if ([string]::IsNullOrWhiteSpace($ManifestPath)) {
    throw "ManifestPath is required unless -Help is specified."
}

$quickEntryMap = [ordered]@{
    EmbeddingOutput                 = 'embedding_output_first_32'
    FirstLayerAttentionNormOutput   = 'first_layer_attention_norm_output_first_32'
    FirstLayerFeedforwardInput      = 'first_layer_feedforward_input_first_32'
    FirstLayerFeedforwardNormOutput = 'first_layer_feedforward_norm_output_first_32'
    FirstLayerFeedforwardOutput     = 'first_layer_feedforward_output_first_32'
    FirstLayerFeedforwardSubNormOutput = 'first_layer_feedforward_sub_norm_output_first_32'
    FirstLayerFeedforwardDownOutput = 'first_layer_feedforward_down_output_first_32'
    FirstLayerFeedforwardOutputRuntimeSemantic = 'first_layer_feedforward_output_runtime_semantic_first_32'
    FinalNormOutput                 = 'final_norm_output_first_32'
    LmHeadOutputLogits              = 'lm_head_output_logits_first_32'
    FirstLayerAttentionOutputFullFirst64        = 'first_layer_attention_output_full_first_64'
    FirstLayerFeedforwardInputFullFirst64       = 'first_layer_feedforward_input_full_first_64'
    FirstLayerFeedforwardNormOutputFullFirst64  = 'first_layer_feedforward_norm_output_full_first_64'
    FirstLayerFeedforwardOutputFullFirst64      = 'first_layer_feedforward_output_full_first_64'
    FirstLayerFeedforwardSubNormOutputFullFirst64 = 'first_layer_feedforward_sub_norm_output_full_first_64'
    FirstLayerFeedforwardDownOutputFullFirst64  = 'first_layer_feedforward_down_output_full_first_64'
    FirstLayerFeedforwardOutputRuntimeSemanticFullFirst64 = 'first_layer_feedforward_output_runtime_semantic_full_first_64'
    FirstLayerFeedforwardInputContext = 'first_layer_feedforward_input_context'
    FirstLayerFeedforwardOutputContext = 'first_layer_feedforward_output_context'
    FirstLayerFeedforwardSubNormOutputContext = 'first_layer_feedforward_sub_norm_output_context'
    FirstLayerFeedforwardDownOutputContext = 'first_layer_feedforward_down_output_context'
    FirstLayerFeedforwardOutputRuntimeSemanticContext = 'first_layer_feedforward_output_runtime_semantic_context'
}

$selectedQuickEntries = @()
foreach ($pair in $quickEntryMap.GetEnumerator()) {
    if (Get-Variable -Name $pair.Key -ValueOnly) {
        $selectedQuickEntries += $pair.Value
    }
}

if ($selectedQuickEntries.Count -gt 1) {
    throw "Specify at most one quick entry switch at a time."
}

if ($selectedQuickEntries.Count -eq 1) {
    if (-not [string]::IsNullOrWhiteSpace($EntryName)) {
        throw "Do not combine -EntryName with a quick entry switch."
    }
    $EntryName = $selectedQuickEntries[0]
}

function Get-HalfValue {
    param(
        [Parameter(Mandatory = $true)]
        [UInt16]$Bits
    )

    if ([System.BitConverter].GetMethod('UInt16BitsToHalf', [type[]]@([UInt16])) -ne $null) {
        return [float][System.BitConverter]::UInt16BitsToHalf($Bits)
    }

    $sign = if (($Bits -band 0x8000) -ne 0) { -1.0 } else { 1.0 }
    $exp = ($Bits -shr 10) -band 0x1f
    $frac = $Bits -band 0x03ff

    if ($exp -eq 0) {
        if ($frac -eq 0) {
            return [float]($sign * 0.0)
        }
        return [float]($sign * [Math]::Pow(2.0, -14) * ($frac / 1024.0))
    }

    if ($exp -eq 31) {
        if ($frac -eq 0) {
            if ($sign -lt 0) {
                return [float]::NegativeInfinity
            }
            return [float]::PositiveInfinity
        }
        return [float]::NaN
    }

    return [float]($sign * [Math]::Pow(2.0, $exp - 15) * (1.0 + ($frac / 1024.0)))
}

function Read-F32Values {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    if (($Bytes.Length % 4) -ne 0) {
        throw "F32 byte length must be a multiple of 4, actual: $($Bytes.Length)"
    }

    $values = New-Object 'System.Collections.Generic.List[float]'
    for ($index = 0; $index -lt $Bytes.Length; $index += 4) {
        $values.Add([System.BitConverter]::ToSingle($Bytes, $index))
    }
    return $values.ToArray()
}

function Read-F16Values {
    param(
        [Parameter(Mandatory = $true)]
        [byte[]]$Bytes
    )

    if (($Bytes.Length % 2) -ne 0) {
        throw "F16 byte length must be a multiple of 2, actual: $($Bytes.Length)"
    }

    $values = New-Object 'System.Collections.Generic.List[float]'
    for ($index = 0; $index -lt $Bytes.Length; $index += 2) {
        $bits = [System.BitConverter]::ToUInt16($Bytes, $index)
        $values.Add((Get-HalfValue -Bits $bits))
    }
    return $values.ToArray()
}

function Get-EntryValues {
    param(
        [Parameter(Mandatory = $true)]
        $Entry,

        [Parameter(Mandatory = $true)]
        [byte[]]$AllBytes
    )

    if ($Entry.dtype -eq 'inline_json') {
        return $Entry.inline_data
    }

    $offset = [int64]$Entry.offset
    $byteLength = [int64]$Entry.byte_length

    if ($offset -lt 0 -or $byteLength -lt 0 -or ($offset + $byteLength) -gt $AllBytes.Length) {
        throw "Entry '$($Entry.Name)' has an out-of-range offset/byte_length"
    }

    $slice = New-Object byte[] $byteLength
    [Array]::Copy($AllBytes, $offset, $slice, 0, $byteLength)

    switch ($Entry.dtype) {
        'F32' { return Read-F32Values -Bytes $slice }
        'F16' { return Read-F16Values -Bytes $slice }
        default { throw "Unsupported dtype '$($Entry.dtype)' in entry '$($Entry.Name)'" }
    }
}

function Resolve-DataBinPath {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$ManifestFilePath
    )

    $generatedData = $Manifest.generated_files | Where-Object { $_.Name -eq 'data.bin' } | Select-Object -First 1
    if ($null -ne $generatedData -and -not [string]::IsNullOrWhiteSpace($generatedData.path)) {
        return $generatedData.path
    }

    return Join-Path -Path (Split-Path -Path $ManifestFilePath -Parent) -ChildPath 'data.bin'
}

$manifestFullPath = (Resolve-Path -Path $ManifestPath).Path
$manifest = Get-Content -Path $manifestFullPath -Raw | ConvertFrom-Json

if ($ListOnly -or [string]::IsNullOrWhiteSpace($EntryName)) {
    $manifest.entries | Select-Object Name, category, layer_index, step_name, tensor_name, dtype, shape, offset, byte_length, element_count
    return
}

$entry = $manifest.entries | Where-Object { $_.Name -eq $EntryName } | Select-Object -First 1
if ($null -eq $entry) {
    throw "Entry not found: $EntryName"
}

$dataBinPath = Resolve-DataBinPath -Manifest $manifest -ManifestFilePath $manifestFullPath
$allBytes = [System.IO.File]::ReadAllBytes($dataBinPath)
$decoded = Get-EntryValues -Entry $entry -AllBytes $allBytes

$result = [pscustomobject]@{
    manifest_path = $manifestFullPath
    data_bin_path = $dataBinPath
    entry = $entry
    decoded_values = $decoded
}

if ($AsJson) {
    $result | ConvertTo-Json -Depth 8
} else {
    $result
}