# Wrapper-only TVM-FFI diagnostic

The frozen native subtile producer/helper, tile and arithmetic remain unchanged. Only compile_bmm replaces make_fake_stream() with make_fake_stream(use_tvm_ffi_env_stream=True) and adds options="--enable-tvm-ffi" to cute.compile. Compile-time descriptors are prepared once as before; the resulting runtime callable takes three Torch tensors directly, without an explicit stream argument or per-call input DLPack conversion.

GPU0 PID414931 first measured ~7.5us DLPack conversion and ~33us ordinary host submission. This passed the explicit5us host-overhead threshold before FFI compilation/testing. Counterbalanced warm host medians were36.511us prior versus6.721us FFI. GPU1 PID415061 confirmation gave35.854us versus6.805us. Host timing measures eight asynchronous submissions after synchronization, excluding the subsequent synchronization; these are CPU dispatch costs, not end-to-end kernel latency.

M16331,N19456,K2560, synthetic BF16 inputs. Same-kernel outputs passed bitwise comparison,256-row/unused-half NaN canaries, nondefault-stream execution and changed-input CUDA graph replay. Graphs, tensors, descriptors and callables remain retained. Exact scripts and nonempty graph DOTs are preserved in the raw archive. The arithmetic helper is unchanged SHA68b7746ce1e6a785124cf2fc86d8de732e754419874c88a2c26b2d221a794d6f.

Initial GPU0 event times were unstable during warmup; no GPU kernel speedup is claimed. GPU1 repeated after200 common replays and measured effectively equal ~1.29ms GPU-event medians. This wrapper-only test uses one fixed weight and does not replace the preceding rotating-weight kernel benchmark. Approximately29us host work saved per call is a model-integration lead, not36×29us guaranteed critical-path improvement: CPU/GPU work can overlap.

Frozen module /root/qvl/experiments/native-epilogue-ffi/native_fused_ffi.py SHA0a0897443d7a4f41361ef24a030ea6356dd6e406d510d5c3f419542bfaf19b30. Direct invocation: compiled(x3d, weight_transposed3d, output3d); no stream argument. Parent/profile received this frozen API for a separate model gate. No model/serving run or production edits were performed here. Both workers finished; no further GPU jobs.

Exact .py sources are provided as .py.txt and unmodified in raw_evidence.tar.gz, alongside reports/logs/graphs and hashes. Pinned /root/qvl/venv-sgl Python, PYTHONPATH=/root/qvl/sglang-prefix-production/python, GPU0/1. No new tile, precision change or kernel algorithm was introduced.
