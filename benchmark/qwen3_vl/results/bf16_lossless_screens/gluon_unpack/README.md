# Explicit-layout lossless reader

Standalone Gluon reconstruction directly in DotOperandLayout eliminates the
prior shared-memory format conversion. Exhaustive65536BF16patterns,random
bits,Gaussian values and six real K/V captures decode bitwise. No production
source changes. Resource probe uses MMA v2 and warps_per_cta=[1,4].

Full attention at BN32,4warps,8splits: packed168registers,1024bytes shared,
zero spills; raw80registers,1024bytes shared. Prior pair reconstruction used
145registers,155648bytes shared at BN128/8warps; these are different schedules,
so reduced shared demand is not a measured occupancy guarantee.

The full raw/packed outputs match bitwise. Synthetic TRT maxabs0.00048828125;
real layer17 maxabs0.0078125 passes atol0.002/rtol0.02. Softmax probability
rounding and reduction follow the prior exact-BF16-input attention prototype;
TRT need not be bitwise because reduction order differs.

Real layer17 K/V repeated across128requests, L8192, Q32/KV8/D128, page32.
Logical cache4GiB exceeds L2; every decode scans it. Three alternating graph
measurements: packed median3.167218ms, raw Gluon1.202456ms, TRT0.590701ms.
The reconstructed format and all KV bits remain unchanged. Despite removing
shared conversion, direct operand gathers and reconstruction remain expensive.
Rejected; no tuning sweep or model integration.

PTX retained remotely in `/root/qvl/experiments/gluon-lossless/`.
Scripts reference previous experimental packer/reduction and captured tensors.
