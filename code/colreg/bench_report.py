import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bench_synth as B
import sys
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "bench_out/bench.json"))
MS = [(m, c) for m, c in (("v1", "tab:red"), ("v2", "tab:green"), ("v3", "tab:blue"), ("v3both", "tab:cyan"), ("steger", "tab:purple")) if m in R["width"][0]]
axes = list(B.AXES)
def lab(a, i):
    d = B.AXES[a][i]; return ",".join(str(v)[:5] for v in d.values())
fig, ax = plt.subplots(4, 4, figsize=(20, 11))
for a_, name in zip(ax.ravel(), axes):
    xs = range(len(R[name]))
    for m, c in MS:
        a_.plot(xs, [max(R[name][i][m]["gross5"] if R[name][i][m]["false_out"] is None else max(R[name][i][m]["gross5"], R[name][i][m]["false_out"]), 1e-3) for i in xs], "o-", c=c, label=m)
    a_.set_yscale("log"); a_.set_ylim(5e-4, 1.2); a_.set_xticks(list(xs)); a_.set_xticklabels([lab(name, i) for i in xs], fontsize=7)
    a_.set_title(name); a_.grid(alpha=.3)
ax[0, 0].legend()
fig.suptitle("synthetic benchmark: failure rate = max(gross>5px or missing rows, false-output rate)  (log; floor 1e-3)")
plt.tight_layout(); plt.savefig("bench_out/curves_fail.png", dpi=80)
fig, ax = plt.subplots(4, 4, figsize=(20, 11))
for a_, name in zip(ax.ravel(), axes):
    for m, c in MS:
        a_.plot(range(len(R[name])), [R[name][i][m]["median"] if R[name][i][m]["median"] is not None else float("nan") for i in range(len(R[name]))], "o-", c=c, label=m)
    a_.set_yscale("log"); a_.set_xticks(range(len(R[name]))); a_.set_xticklabels([lab(name, i) for i in range(len(R[name]))], fontsize=7); a_.set_title(name); a_.grid(alpha=.3)
ax[0, 0].legend(); fig.suptitle("median |error| on GT-present rows (px, best-of-row)")
plt.tight_layout(); plt.savefig("bench_out/curves_median.png", dpi=80)
# markdown table
with open("bench_out/table.md", "w") as f:
    f.write("| axis | level | " + " | ".join(m + " med/miss+bad/false" for m, _ in MS) + " |\n|--|--|" + "--|" * len(MS) + "\n")
    for name in axes:
        for i in range(len(R[name])):
            r = R[name][i]
            f.write(f"| {name} | {lab(name, i)} | " + " | ".join(f"{r[m]['median']:.2f} / {r[m]['gross5']*100:.1f}% / {r[m]['false_out']*100:.1f}%" if r[m]['median'] is not None else "n/a" for m, _ in MS) + " |\n")
print(open("bench_out/table.md").read())
print({m: R["empty"][0][m] for m in R["empty"][0]})
