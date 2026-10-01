# Fixed INT8 LMS route: measured child-process runtime

Two DGX Spark hosts, unchanged model/precision/LoRA, seed 20260929, 2 Turbo + 10 LMS steps, 768×1280 first pass and 1152×1920 second pass. A uses native routing; B uses the fixed eligible cuBLAS route. Both include the same eight-hash evidence cost.

The primary metric is head child start-to-exit monotonic runtime, including initialization, model loading, reference encoding, sampling, decoding, saving and hashes. It excludes parent ComfyUI queue time and network submission. Each sample uses a fresh process and separate compilation directory; OS caches are not flushed. This is neither a cold-disk benchmark nor a resident-model warm benchmark.

| Fixture / pair | Native A, seconds | Candidate B, seconds | Time reduction |
|---|---:|---:|---:|
| 124 frames / 1 | 989.084856 | 877.631478 | 11.268333% |
| 124 frames / 2 | 988.457901 | 882.158044 | 10.754111% |
| 124 frames / 3 | 982.020454 | 886.894957 | 9.686712% |
| 345 frames / first pair | 4701.332134 | 4388.425325 | 6.655705% |

Native output is 24 fps. The long reference is 14.033333 seconds; generated native frames include model alignment padding. These native canvases are not described as an exact 1080×1920 delivery.

Short runs followed the frozen order A1/B1/B2/A2/A3/B3, paired by number. All samples are retained. Median paired reduction is **10.754111%**; median A/B runtimes are 988.457901/882.158044 seconds. Within-group range was about 0.715% for A and 1.050% for B. The third pair is below 10%; n=3 does not establish a population confidence bound. Worker median paired reduction was 10.615669%.

The long pair saved **312.906809 seconds** (about 5 minutes 13 seconds). The LMS stage fell from 4115.602637 to 3828.125048 seconds. Remaining long repetitions were not launched after the first result missed the original 10% target; the owner subsequently accepted the measured benefit. There is no long-video paired median, noise estimate or universal guarantee.

All six short samples and both long samples completed on both hosts with exit 0. Every candidate covered 500 LMS blocks / 4500 eligible calls per host. Eight sampler/floating-pixel hashes, shapes and finite checks matched their respective native oracle. Both output passes passed ffprobe and complete `ffmpeg -xerror` decode; full decoded-video pixel hashes and recovered file hashes matched. The long oracle matched the original successful long production output. Short and long oracles were never mixed.

Long-run host-memory observations were sampled every 30 seconds, not CUDA/VMM peaks. Head child maximum observed swap was 6.578 GiB native and 5.309 GiB candidate; worker child swap was 0. Host OOM-kill counters did not increase. Sampling can miss transients and host counters are not uniquely attributable to the run. The difference in swap is not established as a causal explanation of the speedup or a guarantee of memory stability.

The performance fixtures contain no audio. Exact pixels demonstrate preservation of their existing images, including existing defects, not improved action or identity quality. Audio-bearing production-entry checks and cancellation checks are separate functional evidence, never added to these performance samples.

## Deployed production-entry check

A separate unchanged 56-frame, 24 fps audio-bearing author fixture completed through the normal ComfyUI API on both hosts. Both passes (768×448 and 1152×672) passed all eight native tensor comparisons, full decode and exact decoded video **and audio** comparisons. Each rank recorded 500 LMS blocks and 4500 candidate calls. An actual cancellation of another precisely identified parent also interrupted its worker and cleaned both children. The existing service processes stayed running and `timeout_s` stayed 0. [Machine-readable functional results](../results/2026-10-01/s02_functional.json).

For media comparison, the saved original files were first checked against their original file SHA, then decoded alongside the candidate with identical commands on the same host. An initial comparison to previously stored audio hashes was not equivalent: the unchanged original files themselves produce different audio hashes under the current decoding environment/format. The corrected comparison remained exact, with no tolerance change and no generation retry. This functional run is excluded from all paired speed results.
