# Hardware compression with reversible byte planes

GPU7 standalone, no production edits. Prior hardware experiment used native
BF16 only. This experiment stores sign+mantissa in one byte plane and the full
8-bit exponent in another, retaining all16bits and4GiB allocation capacity.
Actual VMM compressionType1 is verified per handle, with type0 same-layout
control. Exhaustive65536patterns and full real-cache roundtrips match bitwise;
integer checksum outputs also match. A signed-source checksum cast was fixed
before timing; no BF16 reconstruction error occurred.

Four alternating graph measurements,4GiB real values repeated to exceed L2:

| Data | Native compressed ms | Planes compressed ms | Planes plain ms | Conversion ms |
|---|---:|---:|---:|---:|
|K0|0.600058|0.673693|0.672992|1.261158|
|K17|0.600170|0.671818|0.672688|1.258074|
|V17|0.600387|0.665293|0.672400|1.246637|

Read timings include exact bit reconstruction and checksum reduction; conversion
reads native input and writes both planes. Hardware compression helps the V
plane variant about1%, but even that variant is about11% slower than native.
No material gain and no attention integration warranted. This does not measure
hardware compressed byte counts; it measures end-to-end reader latency.
