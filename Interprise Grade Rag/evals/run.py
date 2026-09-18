"""Run from the application directory: python -m evals.run --help."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from dotenv import load_dotenv
from evals.pipeline import load_golden_dataset, run_pipeline, save_results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Capture live RAG outputs or score saved outputs with RAGAS.")
    parser.add_argument("--engine", choices=["hosted", "ragas"], default="hosted")
    parser.add_argument("--limit", type=int, choices=range(1, 16), default=3)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--input", type=Path, help="Saved hosted report or pipeline JSON; avoids new application queries.")
    parser.add_argument("--output", type=Path, default=Path("evals/results/local.json"))
    args = parser.parse_args()
    if args.engine == "hosted":
        if args.input:
            parser.error("--input is for the RAGAS scorer")
        from evals.hosted import run_evaluation
        report = run_evaluation(args.limit, args.compare_baseline)
    else:
        from evals.metrics import run_all_metrics
        if args.compare_baseline and not args.input:
            parser.error("Capture a hosted comparison first and pass its --input report for RAGAS scoring.")
        if args.input:
            source = json.loads(args.input.read_text(encoding="utf-8"))
        else:
            source = load_golden_dataset()
            source["rag_samples"] = source["rag_samples"][:args.limit]
            source = run_pipeline(source)
        if "runs" in source:
            report = {"method": "ragas-0.4.3", "source_run_id": source.get("run_id"), "runs": {}}
            for mode, samples in source["runs"].items():
                report["runs"][mode] = asyncio.run(run_all_metrics({"rag_samples": samples}, print))
        else:
            report = asyncio.run(run_all_metrics(source, print))
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_results(report, args.output)
    print("Saved " + str(args.output))


if __name__ == "__main__":
    main()
