# Data-free telemetry investigation

The investigation deliberately read no competition assets, model checkpoints, labels, tags or predictions. No further match-inference attempt was made.

Findings:

- The bundled PowerShell 7.6.9 includes Microsoft.ApplicationInsights.dll. That assembly contains dc.services.visualstudio.com. This establishes a possible emitter, not attribution of the rejected request.
- The inspected Python environment has no top-level telemetry/Application Insights packages, and a source scan of project modules found no network or telemetry calls. These are limited static checks, not proof that every native dependency is network-free.
- Synthetic inference imported NumPy, PyTorch and project modules, created a randomly initialized tiny network, ran chunked inference, neutral tag fusion and sequence decoding successfully.
- Python auditing recorded no socket-connect, DNS or sendto events. Three subprocess attempts were blocked. A second synthetic run captured caller stacks: all three originated in Python standard-library platform._win32_ver -> _syscmd_ver -> subprocess.check_output. They were OS-version queries, not evidence of telemetry.
- A point-in-time established-TCP query returned no matching Python/PowerShell connections. This can miss short-lived connections and is not packet capture.
- Both synthetic probes completed successfully. The audit hook cannot observe every native networking path or activity in the host application. No actual rejected payload was captured, so its contents and emitting process remain unknown.

Conclusion: the Python guard's subprocess events have been explained, but the external telemetry blocker is unresolved. Do not infer that PyTorch or match data caused the telemetry, and do not claim that the shell opt-out solved it. Preserve v2 and do not retry match inference by switching execution routes. The remaining prerequisite is attribution of the request or verified host-level network isolation approved for restricted-data processing. A Python-only guard is insufficient evidence of that isolation.
