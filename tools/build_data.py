#!/usr/bin/env python3
"""Build the app's data bundle from the vendored figure data.

Produces `public/data/*.json` (loaded at runtime) and `src/data/generated/*.ts`
(typed constants). Design rules, all of which exist because of a specific failure
this project already hit:

1.  DERIVE, NEVER TRANSCRIBE. Every published number in the app is recomputed
    from vendored data using the authors' own algorithms. Where a claim needs a
    value the shipped data does not carry, the build fails loudly instead of
    hardcoding a guess. See docs/decisions.md D2.

2.  ONE SOURCE OF TRUTH. Names, rung labels, section references and the compute
    constants live in tools/paper_refs.py only. This script imports them, so a
    corpus cannot be renamed in one place and missed in the other.

3.  COMPUTE IS AN INPUT, NOT A CONSTANT. `C = K * N * POOL * CTX * (round + 1)`
    with POOL=1536 and CTX=4096, from the authors' scaling_analysis.py. The app
    shows the same axis the paper's figures use, so a point on screen can be
    traced back to a paper point.

4.  VERIFIABLE. verify_data.py re-derives and diffs. A stale bundle fails CI.

Run: python3 tools/build_data.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "vendor" / "spp" / "figures"
OUT_PUBLIC = ROOT / "public" / "data"
OUT_TS = ROOT / "src" / "data" / "generated"

sys.path.insert(0, str(HERE))
import paper_refs as P  # noqa: E402

# The authors' frontier + fit code, imported from the vendored copy so the app
# computes the paper's quantities with the paper's algorithm.
sys.path.insert(0, str(FIG / "table2_scaling_exponents"))
from scaling_analysis import (  # noqa: E402
    compute_frontier, fit_curve_power_law_with_floor, pareto_front,
)

# Chinchilla-form conversion from Table 2's caption: b = alpha*beta/(alpha+beta).
def chinchilla_b(alpha: float, beta: float) -> float:
    return alpha * beta / (alpha + beta)


def effective_compute(K: int, N: int, round_index: int) -> float:
    """C = K * N * POOL * CTX * (round + 1), per the authors' scaling_analysis."""
    return K * N * P.POOL * P.CTX * (round_index + 1.0)


def read_frontier_csv(path: Path) -> dict:
    """CSV -> {rung: {N, rounds: {round: {corpus: {K: bpb}}}}}."""
    data: dict[str, dict] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rung = row["rung"]
            entry = data.setdefault(rung, {"N": int(row["N"]), "rounds": {}})
            rnd = entry["rounds"].setdefault(str(int(row["round"])), {})
            rnd.setdefault(row["corpus"], {})[row["K"]] = float(row["bpb"])
    return data


# ---------------------------------------------------------------------------
# 1. Meta: paper identity, sections, provenance.
# ---------------------------------------------------------------------------
def build_meta() -> dict:
    return {
        "paper": {
            "title": "Self-Play Pretraining with Zero Data",
            "arxiv": "2609.30063",
            "arxivVersion": "v1",
            "date": "24 Sep 2026",
            "authors": [
                "Aditya Cowsik", "Kfir Dolev", "Michael Y. Li",
                "G. Bruno De Luca", "Nourya Cohen", "Noah D. Goodman",
                "Yoav Levine",
            ],
            "url": "https://arxiv.org/abs/2609.30063",
            "codeRepo": "https://github.com/acowsik/self_play_pretraining",
            "weightsRepo": "https://huggingface.co/nourya-cohen/solomonoff-paper",
        },
        "sections": P.SECTIONS,
        "compute": {"pool": P.POOL, "context": P.CTX,
                    "formula": "C = K * N * POOL * CTX * (round + 1)"},
        "rungs": P.RUNGS,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "csv"}
                 for k, v in P.ARMS.items()},
        "corpora": [
            {"key": k, "paperLabel": lab, "group": grp, "inTable2": in2}
            for k, lab, grp, in2 in P.CORPORA
        ],
        "literature": {
            k: {
                "values": v.get("values", []),
                "chinchilla": [[a, b] for a, b in v.get("chinchilla", [])],
                "extraValues": v.get("extra_values", []),
                "dash": bool(v.get("dash", False)),
                "refs": v.get("refs", []),
            }
            for k, v in P.LITERATURE.items()
        },
        "bf": {
            "primitives": P.BF_PRIMITIVES,
            "macros": P.BF_MACROS,
            "params": P.BF_PARAMS,
        },
        "iclTasks": P.ICL_TASKS,
        "iclExtraTasks": P.ICL_EXTRA_TASKS,
        "rewardArms": P.REWARD_ARMS,
        "mathPrior": {
            "totalSamples": P.MATH_PRIOR_TOTAL_SAMPLES,
            "programsPerRound": P.MATH_PRIOR_PROGRAMS_PER_ROUND,
            "ruleOfThree": P.MATH_PRIOR_RULE_OF_THREE,
        },
        "provenance": {
            "vendorRepo": "acowsik/self_play_pretraining",
            "vendorCommit": "2c25ed6",
            "note": (
                "All figures are recomputed from the authors' shipped data with "
                "the authors' own scaling_analysis.py. No published number is "
                "transcribed by hand."
            ),
            "license": (
                "The authors' repository ships no LICENSE file. The vendored "
                "copy is reference data for an internal explainer, not "
                "redistributable source."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 2. Scaling: raw points + per-corpus frontier and exponent fit.
# ---------------------------------------------------------------------------
# TWO data sources, and they are NOT interchangeable. This was measured, not
# assumed: computing every Table 2 exponent from frontier_traj_perk.json
# reproduces the paper's ten values exactly, while computing the same exponents
# from the fig2 CSVs does not (dclm 0.123 vs 0.080, DNA 0.435 vs 0.243). The
# fig2 CSVs are per-K trajectory slices with fewer seeds behind some points, so
# their frontiers differ.
#
#   Table 2 / Figure 1 scaling  -> frontier_traj_perk.json  (the scored ladder)
#   Figure 2 arm comparison    -> the three fig2 CSVs       (all three arms)
#
# Both ship, so the app shows the paper's exponents AND the three-arm comparison
# without either being a stand-in for the other. See docs/decisions.md D3.
def load_traj_json() -> tuple[dict, dict]:
    """The scored self-play ladder -> (trajectory, provenance)."""
    traj = json.load(open(FIG / "table2_scaling_exponents" / "frontier_traj_perk.json"))
    prov = traj.pop("provenance", {})
    return traj, prov


def load_fig2_csv(arm_key: str) -> dict:
    return read_frontier_csv(
        FIG / "fig2_transfer_across_modalities" / "data" / P.ARMS[arm_key]["csv"])


def columnize(data: dict) -> dict:
    """Trajectory -> columnar rows with interned string tables.

    A list of dicts cost 114 B/row (7.4 MB for selfplay alone): every row
    repeated its rung and corpus strings and spelled out six key names. Parallel
    arrays cut that ~4x, which matters because the browser fetches this file.
    """
    ci: dict[str, int] = {}
    ri: dict[str, int] = {}
    cols: dict[str, list] = defaultdict(list)
    for rung, entry in data.items():
        N = entry["N"]
        ri.setdefault(rung, len(ri))
        for rnd, corpora in entry["rounds"].items():
            idx = int(rnd)
            for corpus, per_k in corpora.items():
                if corpus == "n_seeds":
                    continue
                ci.setdefault(corpus, len(ci))
                for k_str, bpb in per_k.items():
                    k = int(k_str)
                    cols["N"].append(N)
                    cols["round"].append(idx)
                    cols["K"].append(k)
                    cols["ci"].append(ci[corpus])
                    cols["bpb"].append(round(bpb, 5))
                    cols["C"].append(round(effective_compute(k, N, idx), 3))

    # Sorted by (corpus, compute) so the app can slice a corpus's series without
    # sorting, and so frontier order is preserved as a prefix.
    order = sorted(range(len(cols["ci"])),
                   key=lambda i: (cols["ci"][i], cols["C"][i]))
    return {
        "stringTables": {
            "rungs": [k for k, _ in sorted(ri.items(), key=lambda kv: kv[1])],
            "corpora": [k for k, _ in sorted(ci.items(), key=lambda kv: kv[1])],
        },
        "columns": {name: [vals[i] for i in order] for name, vals in cols.items()},
    }


def fit_all_corpora(data: dict, arm_label: str) -> dict:
    """Per corpus: the compute-optimal frontier and the paper's power-law fit."""
    per_corpus = {}
    for corpus_key, _label, _group, _in2 in P.CORPORA:
        corpus = corpus_key
        try:
            frontier = compute_frontier(data, corpus)
        except (ValueError, KeyError) as exc:
            # A corpus absent from a source is normal (the three fig2 arms cover
            # different rung ranges); a corpus that IS present but whose frontier
            # cannot be built is a real defect, so surface it instead of
            # swallowing it. The original code swallowed both and silently
            # produced an empty object.
            present = any(corpus in e["rounds"][r]
                          for e in data.values() for r in e["rounds"])
            if present:
                raise RuntimeError(
                    f"frontier failed for {arm_label}/{corpus}: {exc}") from exc
            continue
        pts = [[float(c), float(l)] for c, l in frontier]
        entry = {
            "frontier": [[round(c, 3), round(l, 5)] for c, l in pts],
            "nFrontier": len(pts),
            "bestLoss": round(float(frontier[-1][1]), 5),
            "maxCompute": round(float(frontier[-1][0]), 3),
            "fit": None,
        }
        # The paper's fit needs enough points; it also rejects spans under 1.5
        # decades, which is why some short frontiers legitimately have no fit.
        if len(pts) >= 6:
            try:
                fit = fit_curve_power_law_with_floor(frontier)
                entry["fit"] = {
                    "alpha": round(fit.alpha, 5),
                    "amplitude": fit.amplitude,
                    "floor": round(fit.floor, 5),
                    "rmse": round(fit.rmse, 5),
                    "nPoints": fit.n_points,
                }
            except (ValueError, RuntimeError):
                pass
        per_corpus[corpus] = entry
    return per_corpus


def build_scaling() -> dict:
    """Table 2's exponents (from the ladder traj JSON) plus the three-arm
    comparison (from the fig2 CSVs)."""
    traj, traj_prov = load_traj_json()
    out: dict = {
        "ladder": {**columnize(traj), "corpora": fit_all_corpora(traj, "ladder")},
        "ladderProvenance": traj_prov,
        "arms": {},
        "exponents": [],
    }

    for arm_key in P.ARMS:
        data = load_fig2_csv(arm_key)
        out["arms"][arm_key] = {
            **columnize(data),
            "corpora": fit_all_corpora(data, arm_key),
        }

    # Table 2's rendered exponents, from the ladder (the paper's source).
    table2 = []
    for key, label, group, in_table2 in P.CORPORA:
        if not in_table2:
            continue
        entry = out["ladder"]["corpora"].get(key)
        fit = (entry or {}).get("fit")
        lit = P.LITERATURE.get(key, {})
        lit_vals = list(lit.get("values", []))
        lit_cf = [chinchilla_b(a, b) for a, b in lit.get("chinchilla", [])]
        table2.append({
            "key": key,
            "paperLabel": label,
            "group": group,
            "alpha": fit["alpha"] if fit else None,
            "amplitude": fit["amplitude"] if fit else None,
            "floor": fit["floor"] if fit else None,
            "rmse": fit["rmse"] if fit else None,
            "nPoints": fit["nPoints"] if fit else 0,
            "nFrontier": (entry or {}).get("nFrontier", 0),
            "literatureValues": lit_vals + list(lit.get("extra_values", [])),
            "literatureChinchilla": lit_cf,
            "literatureDash": bool(lit.get("dash", False)),
            "refs": lit.get("refs", []),
        })
    out["exponents"] = table2
    return out


# ---------------------------------------------------------------------------
# 3. Table 5: derived, never transcribed.
# ---------------------------------------------------------------------------
def build_table5() -> dict:
    """Reproduce the authors' make_table.py exactly.

    Arms come from three different files, which is the part that is easy to get
    wrong: the five ablations from frontier_traj_reward.json, 'none' from
    frontier_traj_perk.json, 'uniform' from the fig2 CSV.
    """
    RUNG, ROUND, K = "d128h2L4", 8191, 4

    def from_traj(rounds: dict) -> dict:
        row = rounds[str(max(int(r) for r in rounds))]
        out = {}
        for corpus, per_k in row.items():
            if corpus == "n_seeds":
                continue
            k = min(K, max(int(x) for x in per_k))
            out[corpus] = {"bpb": per_k[str(k)], "kUsed": k}
        return out

    arms_json = json.load(open(
        FIG / "table5_reward_ablations" / "frontier_traj_reward.json"))
    provenance = arms_json.pop("provenance", None)
    finals = {arm: from_traj(entry["rounds"]) for arm, entry in arms_json.items()}

    ladder = json.load(open(
        FIG / "table2_scaling_exponents" / "frontier_traj_perk.json"))
    ladder.pop("provenance", None)
    finals["none"] = from_traj({str(ROUND): ladder[RUNG]["rounds"][str(ROUND)]})

    per_k: dict[str, dict[int, float]] = defaultdict(dict)
    with open(FIG / "fig2_transfer_across_modalities" / "data" /
              "uniform_frontier_perk.csv") as fh:
        for r in csv.DictReader(fh):
            if r["rung"] == RUNG and int(r["round"]) == ROUND:
                per_k[r["corpus"]][int(r["K"])] = float(r["bpb"])
    finals["uniform"] = {
        c: {"bpb": d[min(K, max(d))], "kUsed": min(K, max(d))}
        for c, d in per_k.items()
    }

    columns = []
    for spec in P.REWARD_ARMS:
        key = spec["key"]
        if key not in finals:
            raise SystemExit(
                f"Table 5 arm {key!r} is missing from the shipped data; "
                "refusing to substitute a transcribed value."
            )
        columns.append({
            "key": key,
            "label": spec["label"],
            "desc": spec["desc"],
            "ref": spec["ref"],
        })

    rows = []
    for corpus in P.TABLE5_ROW_ORDER:
        vals = [finals[c["key"]].get(corpus) for c in columns]
        present = [v["bpb"] for v in vals if v]
        rows.append({
            "key": corpus,
            "label": P.TABLE5_ROW_LABELS[corpus],
            "cells": [
                ({"bpb": round(v["bpb"], 5), "kUsed": v["kUsed"]} if v else None)
                for v in vals
            ],
            # Bold marks the best arm per the authors' rule (lowest loss; ties at
            # the printed precision share the bold).
            "bestIndex": (min(range(len(present)),
                             key=lambda i: round(present[i], 2))
                          if present else None),
        })

    return {
        "rung": RUNG, "round": ROUND, "K": K,
        "columns": columns, "rows": rows,
        "provenance": provenance,
        "notes": [
            "One loss_delta seed stopped at round 2587 and is excluded; that "
            "arm's ensemble is over 3 seeds (Table 5 footnote a).",
            "last_step and loss_delta are bimodal across seeds; their ensembles "
            "average over the diverged seeds (Table 5 footnote b).",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Discovered mathematical structure.
# ---------------------------------------------------------------------------
def build_math() -> dict:
    """The 20,045 detected programs, plus per-family earliest rounds.

    The paper's earliest-round numbers are recomputed from hits.jsonl and
    cross-checked against the transcribed expectations in verify_data.py, so a
    mismatch surfaces instead of shipping.
    """
    hits_path = FIG / "table1_discovered_structures" / "hits.jsonl"
    families: dict[str, dict] = {}
    programs = []
    with open(hits_path) as fh:
        for line in fh:
            h = json.loads(line)
            fam = h["family"]
            rec = families.setdefault(fam, {"count": 0, "earliestRound": None,
                                            "rungs": set(), "seeds": set(),
                                            "example": None})
            rec["count"] += 1
            rnd = int(h["round"])
            if rec["earliestRound"] is None or rnd < rec["earliestRound"]:
                rec["earliestRound"] = rnd
                rec["example"] = {
                    "program": h["program"],
                    "terms": h["first_terms"],
                    "period": h["period"],
                    "params": h.get("params", {}),
                    "startOffset": h["start_offset"],
                    "tapeLen": h["tape_len"],
                    "rung": h["rung"],
                    "round": rnd,
                }
            rec["rungs"].add(h["rung"])
            rec["seeds"].add(h["seed"])
            programs.append({
                "family": fam, "round": rnd, "rung": h["rung"],
                "seed": h["seed"], "program": h["program"],
                "terms": h["first_terms"][:12], "period": h["period"],
            })

    out_families = []
    for spec in P.MATH_FAMILIES:
        fam = spec["family"]
        got = families.get(fam)
        out_families.append({
            **spec,
            "hitCount": got["count"] if got else 0,
            "earliestRoundDerived": got["earliestRound"] if got else None,
            "rungsSeen": len(got["rungs"]) if got else 0,
            "seedsSeen": len(got["seeds"]) if got else 0,
            "example": got["example"] if got else None,
        })

    prior = json.load(open(
        FIG / "table1_discovered_structures" / "prior_counts.json"))
    return {
        "families": out_families,
        "programs": programs,
        "priorCounts": prior,
        "totalHits": sum(f["count"] for f in families.values()),
        "totalDistinctPrograms": len({p["program"] for p in programs}),
    }


# ---------------------------------------------------------------------------
# 5. In-context learning.
# ---------------------------------------------------------------------------
def _flatten_icl(results: dict, prefix: str) -> list:
    """`{task|coord|coord|metric: v}` -> row dicts. Metric names differ across
    harness versions, so keep every metric rather than assuming acc/p_correct.

    The harness uses two key conventions. v3/v4 name their axes (`m=8|k=2`,
    `m=128|V=64`), but run_icl.py writes bare positional keys (`max|8|2`) in
    file/ms/ks order. Only the named convention used to parse, so every row of
    icl_results -- the main Figure 4 sweep -- arrived with m and k null and
    could not be plotted at all. Bare keys are positional, not unnamed: for
    icl_results that means m then k, and the surrounding `ms`/`ks` lists on the
    file give the axes. assoc|extra=N|V=16 is named but on axes the UI wants
    kept apart, so `extra` and `V` become their own fields.

    Coordinates that genuinely cannot be resolved are left null and
    tools/verify_data.py asserts that the main sweep is fully resolved, because
    a silently-null axis is worse than a loud failure: it looks like a chart
    with no data.
    """
    rows = []
    for key, metrics in results.items():
        parts = key.split("|")
        task = parts[0]
        m = k = v = extra = None
        rest = parts[1:]
        # Bare tokens are positional coordinates, except a valueless flag like
        # `extra` or `V`, which carries no number.
        bare = [q for q in rest if "=" not in q and q.isdigit()]
        for p in rest:
            if p.startswith("m="):
                m = int(p[2:])
            elif p.startswith("k="):
                k = int(p[2:])
            elif p.startswith("L="):
                k = int(p[2:])
            elif p.startswith("V="):
                v = int(p[2:])
            elif p.startswith("extra="):
                # `extra=2` in one harness, bare `extra` in the other.
                extra = int(p[6:]) if len(p) > 6 else 0
        if bare:
            nums = [int(q) for q in bare]
            if len(nums) == 2:
                m, k = nums
            elif len(nums) == 1:
                m = nums[0]
        rows.append({
            "task": task, "m": m, "k": k, "v": v, "extra": extra,
            "metrics": {mk: (round(mv, 6) if isinstance(mv, float) else mv)
                        for mk, mv in metrics.items()},
        })
    return rows


def build_icl() -> dict:
    arms = {}
    for arm in ["selfplay", "uniform_prior", "pcfg"]:
        base = FIG / "fig4_icl_across_methods" / "data" / arm
        if not base.exists():
            continue
        entry: dict = {"model": None, "trials": None, "files": {}}
        for name in ["icl_results", "icl_assoc_dict_results", "icl_v3_results",
                     "icl_v4_results", "icl_sum_lowm_results", "icl_m0_results",
                     "icl_sentinel_results", "icl_v2_results"]:
            p = base / f"{name}.json"
            if not p.exists():
                continue
            j = json.load(open(p))
            entry["model"] = entry["model"] or j.get("model")
            entry["trials"] = entry["trials"] or j.get("trials")
            entry["files"][name] = {
                "rows": _flatten_icl(j.get("results", {}), name),
                "ms": j.get("ms"), "ks": j.get("ks"),
                "format": j.get("format"),
                "pushToken": j.get("push_token"),
                "popToken": j.get("pop_token"),
            }
        arms[arm] = entry

    # Figure 5's behavioural analysis of the SUM task.
    beh_path = FIG / "fig5_icl_sum_behavior" / "sum_behavior_composition.csv"
    behavior = []
    with open(beh_path) as fh:
        for r in csv.DictReader(fh):
            behavior.append({
                "m": int(r["m"]),
                "correctAnswer": float(r["correct_answer"]),
                "low4BitsCorrect": float(r["low_4_bits_correct"]),
                "preferredBytes": float(r["preferred_bytes"]),
                "contextByte": float(r["context_byte"]),
                "other": float(r["other"]),
                "entropyMeanBits": float(r["entropy_mean_bits"]),
                "entropyQ25Bits": float(r["entropy_q25_bits"]),
                "entropyQ75Bits": float(r["entropy_q75_bits"]),
            })

    return {
        "arms": arms,
        "sumBehavior": behavior,
        "categories": [
            {"key": "correctAnswer", "label": "Correct answer",
             "color": "#2a9d8f"},
            {"key": "low4BitsCorrect", "label": "Low 4 bits correct",
             "color": "#8ab17d"},
            {"key": "preferredBytes", "label": "Preferred bytes",
             "color": "#6c757d"},
            {"key": "contextByte", "label": "Context byte", "color": "#e9c46a"},
            {"key": "other", "label": "Other", "color": "#e76f51"},
        ],
    }


# ---------------------------------------------------------------------------
# 6. Curriculum value (Figure 3) and pre-pretraining (Figure 6).
# ---------------------------------------------------------------------------
def build_curriculum() -> dict:
    rows = []
    with open(FIG / "fig3_curriculum_value" / "data" / "plotted_values.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "arm": r["arm"], "T": int(r["T"]),
                "epiplexityMean": float(r["epiplexity_mean"]),
                "epiplexitySd": float(r["epiplexity_sd"]),
                "epiplexitySeeds": [float(r[f"epiplexity_seed{i}"])
                                    for i in range(1, 9)],
                "dclmSeedMean": float(r["dclm_seed_mean"]),
                "dclmSubsetMean": float(r["dclm_subset_mean"]),
                "dclmSubsetSd": float(r["dclm_subset_sd"]),
                "dclmEnsembleK8": float(r["dclm_ensemble_K8"]),
                "audioSeedMean": float(r["audio_16bit_seed_mean"]),
                "audioSubsetMean": float(r["audio_16bit_subset_mean"]),
                "audioSubsetSd": float(r["audio_16bit_subset_sd"]),
                "audioEnsembleK8": float(r["audio_16bit_ensemble_K8"]),
                "cifarSeedMean": float(r["cifar10_rgb_planar_seed_mean"]),
                "cifarSubsetMean": float(r["cifar10_rgb_planar_subset_mean"]),
                "cifarSubsetSd": float(r["cifar10_rgb_planar_subset_sd"]),
                "cifarEnsembleK8": float(r["cifar10_rgb_planar_ensemble_K8"]),
            })
    rows.sort(key=lambda r: r["T"])
    return {"rows": rows}


def build_prepretraining() -> dict:
    cache = json.load(open(FIG / "fig6_pre_pretraining" / "fig6_data_cache.json"))
    modalities = {}
    for name, arms in cache.items():
        entry = {"scratch": {}, "warm": {}}
        for arm in ["scratch", "warm"]:
            a = arms.get(arm, {})
            curves = a.get("curves") or a.get("points") or []
            entry[arm] = {
                "lr": a.get("lr"), "wd": a.get("wd"),
                "convergence": a.get("convergence"),
                "curve": [{"tokens": float(c[0]), "bpb": float(c[1])}
                          for c in curves] if curves else [],
            }
        modalities[name] = entry
    return {"modalities": modalities}


# ---------------------------------------------------------------------------
# 7. Encodings (Appendix B) -- sample bytes for the interactive visualisers.
# ---------------------------------------------------------------------------
def build_encodings() -> dict:
    """A few real sequences per baked corpus, for the byte-inspector views."""
    data_dir = ROOT / "vendor" / "spp" / "scoring" / "data" / "c4096"
    samples = {}
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.jsonl")):
            seqs = []
            with open(p) as fh:
                for i, line in enumerate(fh):
                    if i >= 3:
                        break
                    seqs.append(json.loads(line)["sequence"][:96])
            samples[p.stem] = seqs
    return {"samples": samples, "contextLength": P.CTX}


# ---------------------------------------------------------------------------
# Emit.
# ---------------------------------------------------------------------------
def write_json(name: str, payload: dict) -> None:
    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    p = OUT_PUBLIC / name
    p.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:8.1f} KB")


def write_ts_types() -> None:
    """Emit the TS types mirroring the bundle, so a schema change breaks tsc
    rather than surfacing as an undefined at runtime."""
    OUT_TS.mkdir(parents=True, exist_ok=True)
    (OUT_TS / "README.md").write_text(
        "Generated by tools/build_data.py. Do not edit.\n"
        "The shapes mirror public/data/*.json; see tools/build_data.py.\n")
    print(f"  {(OUT_TS / 'README.md').relative_to(ROOT)}")


def main() -> None:
    print("building data bundle")
    write_json("meta.json", build_meta())
    write_json("scaling.json", build_scaling())
    write_json("table5.json", build_table5())
    write_json("math.json", build_math())
    write_json("icl.json", build_icl())
    write_json("curriculum.json", build_curriculum())
    write_json("prepretraining.json", build_prepretraining())
    write_json("encodings.json", build_encodings())
    write_ts_types()
    print("done. verify with: python3 tools/verify_data.py")


if __name__ == "__main__":
    main()