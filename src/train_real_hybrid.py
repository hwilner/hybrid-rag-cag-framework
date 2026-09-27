"""Train the real HybridRAGCAG on the Tier-2 dataset with a held-out split.

Protocol: seed-fixed 44-question train split / 11-question held-out split.
Reports held-out token-F1 BEFORE and AFTER training, so the effect of every
training change is measurable and regressions are visible.

Usage:
    python src/train_real_hybrid.py --epochs 2 --lr 1e-5 [--trainable-encoder]
        [--beam-groups 2] [--bfloat16] [--eval-only]
"""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from option3_full_scale_evaluation import create_large_scale_dataset
from hybrid_rag_cag_system import HybridRAGCAGSystem, HybridConfig, MetricsCalculator


def evaluate(system, qa_pairs, beams=None, max_target=None, candidates=None):
    """Mean token-F1 of the full pipeline on the given QA pairs."""
    cfg = system.config
    old = (cfg.num_beams, cfg.max_target_length, cfg.num_candidates)
    if beams is not None:
        cfg.num_beams = beams
    if max_target is not None:
        cfg.max_target_length = max_target
    if candidates is not None:
        cfg.num_candidates = candidates
    scores = []
    for q in qa_pairs:
        pred = system.answer_question(q["question"])
        scores.append(MetricsCalculator.compute_f1(pred, q["answer"]))
    cfg.num_beams, cfg.max_target_length, cfg.num_candidates = old
    return sum(scores) / max(len(scores), 1), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--trainable-encoder", action="store_true")
    parser.add_argument("--beam-groups", type=int, default=1)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--lambda-contrastive", type=float, default=None,
                        help="Override contrastive loss weight (0 disables candidate "
                             "generation during training -- much lighter on CPU).")
    parser.add_argument("--lambda-diversity", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--baseline-f1", type=float, default=None,
                        help="skip eval-before and reuse this measured baseline")
    parser.add_argument("--output", default="/mnt/agents/work/train_real_hybrid_log.json")
    parser.add_argument("--max-train", type=int, default=None,
                        help="cap training questions (shorter run for sandbox windows)")
    parser.add_argument("--max-eval", type=int, default=None,
                        help="cap held-out eval questions")
    parser.add_argument("--eval-beams", type=int, default=1,
                        help="beam width for held-out eval (1 = greedy, ~4x faster)")
    parser.add_argument("--eval-max-target", type=int, default=32,
                        help="max generated tokens during eval (answers are short spans)")
    parser.add_argument("--eval-candidates", type=int, default=2,
                        help="candidate answers per question during eval")
    parser.add_argument("--no-ckpt", action="store_true",
                        help="disable checkpointing (FUSE-mount writes can crash the sandbox)")
    args = parser.parse_args()

    corpus, questions = create_large_scale_dataset()
    rng = random.Random(42)
    idx = list(range(len(questions)))
    rng.shuffle(idx)
    test_idx = set(idx[:11])
    train_qa = [q for i, q in enumerate(questions) if i not in test_idx]
    test_qa = [q for i, q in enumerate(questions) if i in test_idx]
    if args.max_train:
        train_qa = train_qa[: args.max_train]
    if args.max_eval:
        test_qa = test_qa[: args.max_eval]
    print(f"[train] {len(train_qa)} train / {len(test_qa)} held-out questions")

    cfg = HybridConfig()
    cfg.trainable_encoder = args.trainable_encoder
    cfg.num_beam_groups = args.beam_groups
    if args.lambda_contrastive is not None:
        cfg.lambda_contrastive = args.lambda_contrastive
    if args.lambda_diversity is not None:
        cfg.lambda_diversity = args.lambda_diversity
    if args.bfloat16:
        cfg.generator_dtype = "bfloat16"
    system = HybridRAGCAGSystem(cfg)
    system.index_corpus(corpus)
    model = system.model

    log = {"args": vars(args)}

    ckpt_exists = (os.path.exists("/tmp/train_ckpt/meta.pt") or
                   os.path.exists("/mnt/agents/work/train_ckpt/meta.pt"))
    if ckpt_exists and os.path.exists(args.output):
        try:
            prev = json.load(open(args.output))
            log["heldout_f1_before"] = prev["heldout_f1_before"]
            print(f"[train] reused held-out F1 before = {prev['heldout_f1_before']:.3f} "
                  "(checkpoint exists; skipping re-evaluation)", flush=True)
        except Exception:
            ckpt_exists = False
    if "heldout_f1_before" not in log and args.baseline_f1 is not None:
        log["heldout_f1_before"] = args.baseline_f1
        print(f"[train] reused held-out F1 before = {args.baseline_f1:.3f} "
              "(--baseline-f1)", flush=True)
    if "heldout_f1_before" not in log:
        print("[train] evaluating held-out split BEFORE training...", flush=True)
        f1_before, _ = evaluate(system, test_qa, beams=args.eval_beams,
                                max_target=args.eval_max_target,
                                candidates=args.eval_candidates)
        log["heldout_f1_before"] = f1_before
        with open(args.output, "w") as fh:
            json.dump(log, fh, indent=2)
        print(f"[train] held-out F1 before = {f1_before:.3f}", flush=True)

    if not args.eval_only:
        # Memory-capped CPU container: freeze BART's encoder stack and train
        # the decoder only. Grads+momentum for 200M bf16 params instead of
        # 406M -- the difference between fitting and being OOM-killed at 3 GB.
        for p in model.generator.model.model.encoder.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        print(f"[train] trainable parameters: {n_params / 1e6:.1f}M")
        # SGD+momentum keeps one state buffer per parameter; AdamW keeps two
        # and pushes a memory-capped CPU container past its limit.
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

        model.train()
        step = 0
        # Checkpoint locally (fast, safe); a detached copier mirrors shards to
        # the persistent mount -- writing the mount directly from this process
        # gets it killed when the FUSE mount hiccups.
        ckpt_dir = "/tmp/train_ckpt"
        mount_ckpt_dir = "/mnt/agents/work/train_ckpt"
        SHARD = 80 * 1024 * 1024  # keep every file under the mount's ~100MB cap

        def save_ckpt():
            if args.no_ckpt:
                return
            os.makedirs(ckpt_dir, exist_ok=True)
            sd = model.state_dict()
            counts = {}
            for i, (k, v) in enumerate(sd.items()):
                t = v.detach().cpu()
                nbytes = t.numel() * t.element_size()
                nparts = max(1, (nbytes + SHARD - 1) // SHARD)
                rows = (t.shape[0] + nparts - 1) // nparts if t.dim() else 1
                for j in range(nparts):
                    part = t[j * rows:(j + 1) * rows] if t.dim() else t
                    torch.save(part, os.path.join(ckpt_dir, f"m{i:04d}_{j}.pt"))
                counts[k] = nparts
            torch.save({"step": step, "counts": counts}, os.path.join(ckpt_dir, "meta.pt"))
            torch.save(opt.state_dict(), os.path.join(ckpt_dir, "opt.pt"))
            # Detached mirror to the persistent mount as tar chunks <100MB
            # (the mount kills long many-file writes). If the mount hiccups,
            # only the copier dies, never the trainer.
            import subprocess
            os.makedirs(mount_ckpt_dir, exist_ok=True)
            subprocess.Popen(
                'rm -f {d}/ck.tgz.part_*; tar cf - -C /tmp train_ckpt | split -b 80m - {d}/ck.tgz.part_'
                .format(d=mount_ckpt_dir), shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Restore local checkpoint from mount tar chunks if /tmp was wiped.
        if not os.path.exists(os.path.join(ckpt_dir, "meta.pt")):
            parts = sorted(p for p in os.listdir(mount_ckpt_dir)
                           if p.startswith("ck.tgz.part_")) if os.path.isdir(mount_ckpt_dir) else []
            if parts:
                import subprocess
                subprocess.run(
                    'cat {} | tar xf - -C /tmp'.format(
                        " ".join(os.path.join(mount_ckpt_dir, p) for p in parts)),
                    shell=True)
        meta_path = os.path.join(ckpt_dir, "meta.pt")
        if os.path.exists(meta_path):
            try:
                src_dir = os.path.dirname(meta_path)
                meta = torch.load(meta_path, weights_only=False)
                sd = model.state_dict()
                loaded = {}
                for i, k in enumerate(sd):
                    parts = [torch.load(os.path.join(src_dir, f"m{i:04d}_{j}.pt"),
                                        map_location="cpu", weights_only=True)
                             for j in range(meta["counts"][k])]
                    loaded[k] = torch.cat(parts) if len(parts) > 1 else parts[0]
                model.load_state_dict(loaded)
                step = meta["step"]
                opt_p = os.path.join(src_dir, "opt.pt")
                if os.path.exists(opt_p):
                    opt.load_state_dict(torch.load(opt_p, weights_only=False))
                print(f"[train] resumed from checkpoint at step {step}", flush=True)
            except Exception as exc:
                print(f"[train] checkpoint unusable ({exc}); starting fresh", flush=True)
                step = 0

        total_steps = args.epochs * len(train_qa)
        for epoch in range(args.epochs):
            rng.shuffle(train_qa)
            for q in train_qa:
                if step >= total_steps:
                    break
                # Skip steps already done in a previous (interrupted) run.
                epoch_step = step - epoch * len(train_qa)
                if epoch > 0 and epoch_step < 0:
                    pass
                # Retrieve context for this question from the corpus.
                docs = model.retriever.retrieve([q["question"]], k=3)[0][0]
                context = " ".join(docs[:3])
                losses = model.compute_losses([q["question"]], [context], [q["answer"]])
                opt.zero_grad()
                losses["total_loss"].backward()
                opt.step()
                tl, gl, cl = (losses["total_loss"].item(),
                              losses["generation_loss"].item(),
                              losses["contrastive_loss"].item())
                del losses
                import gc, ctypes
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
                step += 1
                print(f"[train] step {step}/{total_steps}  total={tl:.4f} "
                      f"gen={gl:.4f} contrast={cl:.4f}", flush=True)
                if step % 5 == 0:
                    save_ckpt()
                # checkpoint the log so a crash doesn't lose progress
                with open(args.output, "w") as fh:
                    json.dump(log, fh, indent=2)

    print("[train] evaluating held-out split AFTER training...")
    f1_after, _ = evaluate(system, test_qa, beams=args.eval_beams,
                           max_target=args.eval_max_target,
                           candidates=args.eval_candidates)
    log["heldout_f1_after"] = f1_after
    f1_before = log["heldout_f1_before"]
    log["delta"] = f1_after - f1_before
    print(f"[train] held-out F1 after  = {f1_after:.3f}")
    print(f"[train] delta              = {f1_after - f1_before:+.3f}")

    with open(args.output, "w") as fh:
        json.dump(log, fh, indent=2)


if __name__ == "__main__":
    main()
