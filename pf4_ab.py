"""PF-4: two short runs of the real trainer.py, differing only in the recipe knobs.

    uv run python pf4_ab.py A        # control: schedule + selection + decode only
    uv run python pf4_ab.py B        # full change set

Runs trainer.py's actual __main__ body with a handful of config lines substituted, so
what is measured is the file that will be launched for the 2-day run, not a copy of it.
Each arm writes to its own model_path, so neither touches savemodel/.

Read the result as a disaster detector, not a tie-breaker. Ten epochs systematically
favour the control: weaker augmentation and no oversampling both pay off late and cost
early. B being a little behind A at epoch 9 is the expected outcome.
    proceed with B   if map_B >= map_A - 0.010
    fall back to A   if map_B <  map_A - 0.030
"""
import sys

ARM = (sys.argv[1] if len(sys.argv) > 1 else "B").upper()
EPOCHS = 10

# Both arms get the schedule, selection and decode fixes - those are not in question.
COMMON = {
    "epochs = 240": f"epochs = {EPOCHS}",
    # 10% warmup, not the real run's 1.25%: a 10-epoch cosine needs proportionally more
    # ramp, and the point here is that the curve completes, which it does in both arms.
    "warmup_epochs = 3": "warmup_epochs = 1",
    'model_path = "savemodel"': f'model_path = "/tmp/pf4_{ARM}"',
    "deadline_hours = 45.0": "deadline_hours = 6.0",
    "cache_images = True": "cache_images = False",
}

# The control keeps run 1's data recipe so the difference is attributable.
CONTROL = {
    "mosaic_p = 0.2": "mosaic_p = 0.5",
    "cutout_p = 0.2": "cutout_p = 0.5",
    "gauss_min_overlap = 0.7": "gauss_min_overlap = 0.3",
    "rfs_thresh = 0.05": "rfs_thresh = 0",
    "class_weighted_ce = True": "class_weighted_ce = False",
}

# Arm C isolates the repeat-factor sampler: everything in B except RFS. B's deficit vs A
# was concentrated entirely in the most common classes (-0.038 on the top 10, -0.001 on
# the rarest), which is the signature of exposure dilution rather than of the radius or
# the mosaic change - this arm confirms or refutes that.
NO_RFS = {"rfs_thresh = 0.05": "rfs_thresh = 0"}

src = open("trainer.py").read()
edits = {"A": {**COMMON, **CONTROL}, "B": dict(COMMON), "C": {**COMMON, **NO_RFS}}[ARM]
for old, new in edits.items():
    assert src.count(old) == 1, f"substitution target not unique: {old!r}"
    src = src.replace(old, new)

print(f"=== PF-4 arm {ARM} ===")
for old, new in edits.items():
    print(f"    {new}")
print(flush=True)

exec(compile(src, "trainer.py", "exec"), {"__name__": "__main__", "__file__": "trainer.py"})
