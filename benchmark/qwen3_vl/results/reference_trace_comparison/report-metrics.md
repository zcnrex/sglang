# Five-step profiler comparison

These are instrumented GPU traces, not a new serving throughput benchmark or a direct TTFT measurement. All five steps are retained. SGLang measures `ModelRunner.forward`, including graph/attention metadata; vLLM measures nested decoder plus vocabulary projection, with worker bookkeeping and sampling separate. The two scope boundaries are not automatically identical.

## Decode timings

Microseconds; median across five steps. Union counts occupied kernel intervals once; sum counts overlapping kernels repeatedly. Span is first kernel start to last kernel end, not a dependency critical path.

|Framework|B|Kernels/step|Kernel sum|Interval union|Span|Sum − union|
|---|---:|---:|---:|---:|---:|---:|
|sglang|8|294|4041.0|3627.1|3634.4|412.5|
|sglang|16|330|5454.1|5183.7|5194.2|272.0|
|sglang|128|366|25694.5|25368.3|25374.7|326.9|
|vllm|8|510|4045.5|3998.9|4154.1|46.5|
|vllm|16|510|5426.5|5381.5|5542.6|45.0|
|vllm|128|474|25605.2|25575.8|25756.9|28.5|

## Operation counts and kernel duration sums

These sums are attribution, not additive wall-clock savings. QKV fused kernels include GEMM and preparation and cannot be split into separate costs. Generated first-layer variants remain explicitly in Other.

|Framework|B|Operation|Count/step|Median sum µs|
|---|---:|---|---:|---:|
|sglang|8|Attention|36|1776.3|
|sglang|8|Other (metadata/gather/copy)|4|14.6|
|sglang|8|Other GEMM/reduction|109|1391.0|
|sglang|8|QKV GEMM + norm/MRoPE/cache|36|421.2|
|sglang|8|Residual/RMSNorm|73|319.5|
|sglang|8|SiLU × up|36|117.7|
|sglang|16|Attention|36|3056.8|
|sglang|16|Other (metadata/gather/copy)|4|19.6|
|sglang|16|Other GEMM/reduction|145|1778.9|
|sglang|16|QK/MRoPE/cache preparation|36|148.5|
|sglang|16|Residual/RMSNorm|73|332.2|
|sglang|16|SiLU × up|36|121.8|
|sglang|128|Attention|36|22740.1|
|sglang|128|Other (metadata/gather/copy)|4|56.7|
|sglang|128|Other GEMM/reduction|181|2124.1|
|sglang|128|QK/MRoPE/cache preparation|36|210.4|
|sglang|128|Residual/RMSNorm|73|313.9|
|sglang|128|SiLU × up|36|256.1|
|vllm|8|Attention|36|1664.3|
|vllm|8|Other (includes first-layer compiled variants)|43|57.2|
|vllm|8|Other GEMM/reduction|217|1799.0|
|vllm|8|QK/MRoPE/cache preparation|106|249.3|
|vllm|8|Residual/RMSNorm|72|225.6|
|vllm|8|SiLU × up|36|54.2|
|vllm|16|Attention|36|3038.5|
|vllm|16|Other (includes first-layer compiled variants)|43|45.2|
|vllm|16|Other GEMM/reduction|217|1815.2|
|vllm|16|QK/MRoPE/cache preparation|106|246.2|
|vllm|16|Residual/RMSNorm|72|226.7|
|vllm|16|SiLU × up|36|56.6|
|vllm|128|Attention|36|22780.5|
|vllm|128|Other (includes first-layer compiled variants)|43|48.7|
|vllm|128|Other GEMM/reduction|181|2151.4|
|vllm|128|QK/MRoPE/cache preparation|106|296.2|
|vllm|128|Residual/RMSNorm|72|241.7|
|vllm|128|SiLU × up|36|88.1|

## All steps

|Framework|B|Step|Count|Sum µs|Union µs|Span µs|
|---|---:|---:|---:|---:|---:|---:|
|sglang|8|0|294|4044.8|3627.1|4168.2|
|sglang|8|1|294|4038.4|3626.5|3632.3|
|sglang|8|2|294|4035.0|3623.4|3629.4|
|sglang|8|3|294|4041.0|3628.5|3634.4|
|sglang|8|4|294|4041.9|3629.4|3635.4|
|sglang|16|0|330|5453.4|5182.4|5926.8|
|sglang|16|1|330|5454.1|5182.1|5190.1|
|sglang|16|2|330|5465.0|5192.1|5200.0|
|sglang|16|3|330|5460.3|5186.6|5194.2|
|sglang|16|4|330|5453.2|5183.7|5191.9|
|sglang|128|0|366|25703.2|25379.4|25959.8|
|sglang|128|1|366|25708.2|25381.2|25387.5|
|sglang|128|2|366|25692.6|25363.1|25368.9|
|sglang|128|3|366|25694.5|25368.3|25374.7|
|sglang|128|4|366|25686.7|25357.1|25364.1|
|vllm|8|0|510|4057.9|4009.7|4168.1|
|vllm|8|1|510|4047.4|4002.0|4158.5|
|vllm|8|2|510|4045.5|3998.9|4154.1|
|vllm|8|3|510|4038.0|3993.2|4149.7|
|vllm|8|4|510|4042.4|3994.5|4149.2|
|vllm|16|0|510|5422.9|5378.9|5540.7|
|vllm|16|1|510|5426.5|5381.5|5542.6|
|vllm|16|2|510|5429.5|5388.5|5547.4|
|vllm|16|3|510|5431.3|5385.3|5543.8|
|vllm|16|4|510|5425.5|5380.5|5539.1|
|vllm|128|0|474|25602.7|25574.3|25751.1|
|vllm|128|1|474|25598.3|25570.5|25749.7|
|vllm|128|2|474|25615.9|25587.3|25767.8|
|vllm|128|3|474|25614.3|25585.8|25763.3|
|vllm|128|4|474|25605.2|25575.8|25756.9|

## Raw triage and exact-name data

- sglang B8: [exact-name metrics](analysis_inputs/sglang-b8/decode-metrics.json), [correlation windows](analysis_inputs/sglang-b8/decode-windows.json).
  - [Unmodified three-table triage: decode-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b8/decode-TP0.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b8/prefill-TP0.trace.json.gz.triage.txt)
- sglang B16: [exact-name metrics](analysis_inputs/sglang-b16/decode-metrics.json), [correlation windows](analysis_inputs/sglang-b16/decode-windows.json).
  - [Unmodified three-table triage: decode-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b16/decode-TP0.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b16/prefill-TP0.trace.json.gz.triage.txt)
- sglang B128: [exact-name metrics](analysis_inputs/sglang-b128/decode-metrics.json), [correlation windows](analysis_inputs/sglang-b128/decode-windows.json).
  - [Unmodified three-table triage: decode-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b128/decode-TP0.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill-TP0.trace.json.gz.triage.txt](analysis_inputs/sglang-b128/prefill-TP0.trace.json.gz.triage.txt)
- vllm B8: [exact-name metrics](vllm/b8-v2/decode-metrics.json), [correlation windows](vllm/b8-v2/decode-windows.json).
  - [Unmodified three-table triage: prefill_rank0.1791490998474875451.pt.trace.json.gz.triage.txt](vllm/b8-v2/prefill_rank0.1791490998474875451.pt.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill_rank0.1791491002176874222.pt.trace.json.gz.triage.txt](vllm/b8-v2/prefill_rank0.1791491002176874222.pt.trace.json.gz.triage.txt)
- vllm B16: [exact-name metrics](vllm/b16-v2/decode-metrics.json), [correlation windows](vllm/b16-v2/decode-windows.json).
  - [Unmodified three-table triage: prefill_rank0.1791491002446704117.pt.trace.json.gz.triage.txt](vllm/b16-v2/prefill_rank0.1791491002446704117.pt.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill_rank0.1791491008281411705.pt.trace.json.gz.triage.txt](vllm/b16-v2/prefill_rank0.1791491008281411705.pt.trace.json.gz.triage.txt)
- vllm B128: [exact-name metrics](vllm/b128-v2/decode-metrics.json), [correlation windows](vllm/b128-v2/decode-windows.json).
  - [Unmodified three-table triage: prefill_rank0.1791491168898869904.pt.trace.json.gz.triage.txt](vllm/b128-v2/prefill_rank0.1791491168898869904.pt.trace.json.gz.triage.txt)
  - [Unmodified three-table triage: prefill_rank0.1791491199967997927.pt.trace.json.gz.triage.txt](vllm/b128-v2/prefill_rank0.1791491199967997927.pt.trace.json.gz.triage.txt)
