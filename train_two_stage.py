"""Train the class-agnostic detector, then its crop classifier.

Example:
    python train_two_stage.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="data/train_dataset/train_label.json")
    parser.add_argument("--output-dir", default="savemodel_two_stage")
    parser.add_argument("--proposal-backbone", default="swin_t")
    parser.add_argument("--proposal-epochs", type=int, default=120)
    parser.add_argument("--proposal-batch-size", type=int, default=32)
    parser.add_argument("--classifier-epochs", type=int, default=30)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args(argv)


def build_commands(args):
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    proposal_dir = output / "proposal"
    classifier_dir = output / "classifier"
    manifest_dir = output / "cache"
    proposal_weights = proposal_dir / "model.pth"

    proposal = [
        sys.executable, str(ROOT / "trainer_proposal.py"),
        "--annotation", args.annotation,
        "--output-dir", str(proposal_dir),
        "--backbone", args.proposal_backbone,
        "--epochs", str(args.proposal_epochs),
        "--batch-size", str(args.proposal_batch_size),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    classifier = [
        sys.executable, str(ROOT / "trainer_stage2.py"),
        "--annotation", args.annotation,
        "--output-dir", str(classifier_dir),
        "--manifest-dir", str(manifest_dir),
        "--detector-weights", str(proposal_weights),
        "--detector-backbone", args.proposal_backbone,
        "--detector-class-agnostic",
        "--epochs", str(args.classifier_epochs),
        "--batch-size", str(args.classifier_batch_size),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
    ]
    if args.cache_images:
        proposal.append("--cache-images")
    if args.rebuild_manifest:
        classifier.append("--rebuild-manifest")
    if args.no_amp:
        proposal.append("--no-amp")
        classifier.append("--no-amp")
    if args.resume:
        proposal_last = proposal_dir / "last.pth"
        classifier_last = classifier_dir / "last.pth"
        if proposal_last.exists():
            proposal.extend(["--resume", str(proposal_last)])
        if classifier_last.exists():
            classifier.extend(["--resume", str(classifier_last)])
    return proposal, classifier, proposal_weights


def main():
    proposal, classifier, proposal_weights = build_commands(parse_args())
    print("\n==> Stage 1/2: class-agnostic proposal detector", flush=True)
    subprocess.run(proposal, cwd=ROOT, check=True)
    if not proposal_weights.is_file():
        raise SystemExit(f"proposal training did not produce {proposal_weights}")
    print("\n==> Stage 2/2: crop classifier", flush=True)
    subprocess.run(classifier, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
