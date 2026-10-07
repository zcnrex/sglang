# XQA-style swizzled producer gate

External CUDA producer prototype, GPU0. Representative32-row tile,16-byte
grains, XOR-swizzled shared destinations. Follows the row/grain addressing
pattern of XQA copyPartialHeadsAsync, but is not a drop-in instantiation of
all XQA template configurations. Native path uses cp.async; compressed path
loads8sign/mantissa bytes and4exponent bytes to reconstruct16BF16bytes, with
raw fallback using cp.async. Both commit/wait then synchronize the CTA before
shared-memory consumption. No attention kernel or production code changed.

All65536BF16patterns,randombits and six captured real K/V tensors reconstruct
bitwise. Checksum consumption matches exactly. Four compiled variants use
24–26registers,8KiBshared,one barrier,no reported spills.

Repeated real layer17 K rows:4GiB logical cache,97.7036%compressed rows.
Three alternating CUDA-graph rounds: native median0.701863ms, packed1.089389ms
(55.2%slower). Both include shared-memory consumption/checksum output. This
producer gate fails despite low register/shared-memory demand, so full XQA
integration was not attempted. No broad tuning or serving experiment.

Compile with nvcc -O3 -arch=sm_103 -shared -Xcompiler -fPIC
--ptxas-options=-v producer.cu -o producer.so. Binary remains only remotely;
scripts reference `/root/qvl/experiments/xqa-producer/` and prior packer/data.
