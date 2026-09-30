"""Exercise repeated launches and the L2 step loop on one plan instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from .buffers import _external_buffers
from .plan import PlanLibrary


def run(plan: PlanLibrary, batch: int, steps: int, vocab: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if steps < 1 or (plan.info.phase == 1 and
                     plan.info.past_lo + steps - 1 > plan.info.past_hi):
        raise ValueError("smoke steps exceed the plan past interval")
    buffers = _external_buffers(plan, batch, vocab)
    written = {name: tensor for name, tensor in buffers.items()
               if name == "serving.tokens" or name.startswith("kv_cache.")}
    snapshot = {name: value.clone() for name, value in written.items()}
    instance = plan.create(batch, {k: v.data_ptr() for k, v in buffers.items()},
                           torch.cuda.current_device())
    stream = torch.cuda.current_stream()
    arms: dict[str, dict] = {}
    report = {"arms": arms, "mismatches": {}, "watchdog": None, "pass": False}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    manifest = Path(str(plan.path) + ".plan.json")
    metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
    paged = metadata.get("pg") == "pages"
    try:
        instance.set_steps([plan.info.past_lo + i for i in range(steps)]
                           if plan.info.phase == 1 else [plan.info.past_lo])
        for label, mode, loop in (("L2_separate", 2, False),
                                  ("L2_loop", 2, True),
                                  ("L1_separate", 1, False)):
            if loop and (plan.info.phase != 1 or not paged or not metadata.get("paged_la", True) or
                         not hasattr(plan.lib, "tm_plan_launch_steps")):
                continue
            for name, value in written.items():
                value.copy_(snapshot[name])
            stream.synchronize()
            count = steps if plan.info.phase == 1 else 2
            if loop:
                instance.launch_steps(0, count, mode, stream.cuda_stream)
                stream.synchronize()
            else:
                for i in range(count):
                    instance.launch(i if plan.info.phase == 1 else 0,
                                    mode, stream.cuda_stream)
                    stream.synchronize()  # Deliberately reproduces the second-launch failure.
            outputs[label] = {name: value.clone() for name, value in written.items()}
            arms[label] = {"launches": 1 if loop else count, "mode": mode}
        reference = outputs["L2_separate"]
        for label, values in outputs.items():
            report["mismatches"][label] = {
                name: int((value != reference[name]).sum().item())
                for name, value in values.items()}
        report["pass"] = all(n == 0 for arm in report["mismatches"].values()
                             for n in arm.values())
    except BaseException as error:
        report["error"] = repr(error)
        report["watchdog"] = instance.watchdog()
        (out / "watchdog.json").write_text(json.dumps(report["watchdog"], indent=2) + "\n")
    finally:
        (out / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        try:
            instance.close()
        except BaseException:
            pass  # A trapped CUDA context may not be able to free device allocations.
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    vocab = json.loads((args.model / "config.json").read_text())["vocab_size"]
    report = run(PlanLibrary(args.so), args.batch, args.steps, vocab, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, default=str))
    if not report["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
