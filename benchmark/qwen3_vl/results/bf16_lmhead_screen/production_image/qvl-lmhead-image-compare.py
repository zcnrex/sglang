import json
import pathlib

r = pathlib.Path("/root/qvl/experiments/lmhead-production/image")
a = json.load(open(r / "control/responses.json"))
b = json.load(open(r / "candidate/responses.json"))
checks = []
for x, y in zip(a, b):
    c, d = x["choices"][0], y["choices"][0]
    checks.append(
        {
            "content_equal": c["message"]["content"] == d["message"]["content"],
            "logprobs_equal": c["logprobs"] == d["logprobs"],
            "control_tokens": x["usage"]["completion_tokens"],
            "candidate_tokens": y["usage"]["completion_tokens"],
        }
    )
proof = [
    json.loads(l)
    for p in (r / "candidate").glob("lmhead-proof-pid*.jsonl")
    for l in p.read_text().splitlines()
]
assert any(x["kind"] == "graph4_replay" for x in proof)
(r / "comparison.json").write_text(
    json.dumps({"checks": checks, "graph4_replay": True}, indent=2)
)
print(json.dumps(checks))
assert all(x["content_equal"] and x["logprobs_equal"] for x in checks)
