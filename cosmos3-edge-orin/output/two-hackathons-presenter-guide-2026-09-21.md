# Two hackathons, one local vision demo

Use slides 1–8 for the talk. Slides 9–14 answer technical questions. The final live demo is [on the Orin](https://192.168.6.252:8443/).

| Slide | Spoken message | Five-minute pace |
| --- | --- | --- |
| 1 | “We wanted useful local camera understanding on an 8 GB Orin Nano. Two campaigns made different engineering choices.” | 20 sec |
| 2 | “The first campaign used broader INT4 and maximum clocks. This task used selective MLP INT4 at stock 25 W and built a streaming application independently.” | 35 sec |
| 3 | “The screen says 197 milliseconds on the server and 312 in the browser. Both are valid, but they answer different questions.” Show where each number appears. | 40 sec |
| 4 | “Human steering mattered. We corrected the defaults, separated image detail from answer length, fixed the timing boundary and restored the original demo controls.” | 40 sec |
| 5 | “On the same fixed input, original FP16 started text at 249.5 milliseconds and MLP at 239.3. That is about 10 milliseconds on this fixture, with 30 measurements per engine.” | 55 sec |
| 6 | “The first report has useful throughput and memory gains. Its 253 milliseconds means a completed answer; our 239 means first text. These cannot form a speed ranking.” | 40 sec |
| 7 | “The search stopped when improvements became small. The smaller MLP profile saved memory, but failed the original 10% rule. The agent clarified that policy after measurement, and the record preserves that correction.” | 45 sec |
| 8 | “The result is a working device and a clearer measurement contract. NVIDIA supplies the model/runtime foundation. Our contribution is the integration, UI, controls, scoped tuning and evidence.” | 25 sec |

For a seven-minute talk, spend another minute on the screenshot and human steering, and another minute explaining the controlled test and stopping rule. For ten minutes, add a 60-second live demonstration and use the appendix for questions. Turn off live streaming during explanation if the changing captions distract; manually run one inference when ready. Do not run benchmarks during the demo.

Useful exact wording:

- “About 0.24 seconds to **first text** on the controlled fixture,” not to complete a caption.
- “The first campaign **reported** 22.5 to 76 tokens per second and 6.02 to 3.70 GB process RSS.” It was source-reviewed here, not rebenchmarked.
- “Shared system RAM” for the UI memory graph. It includes the OS and applications; it is not dedicated GPU VRAM.
- “The original engine is FP16.” The user’s screenshot filename says BF16; deployment receipts identify FP16.
- “The two branches have not been ranked under identical conditions.” Different power, precision scope, image detail, cache and timers remain.

Appendix: 9 architecture; 10 interventions; 11 user screenshot observations; 12 timing boundaries and unequal answer lengths; 13 final versus intermediate defaults; 14 credits and source provenance. All slides have source notes. One additional logged request started during the MLP measurement, and engine order was sequential. No fixed/marginal cost fit is claimed for the interrupted sweep.
