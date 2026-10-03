"""Shared vocabulary for the playground's data pipeline.

Every dataset name, model rung, and figure reference the app displays is
defined here once, so the TypeScript types and the generated JSON cannot drift
apart. The `*_SECTION` constants are the paper locations that back each claim;
the app cites these, so a wrong or missing section reference is visible in review.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Paper section references. The app shows these next to every claim.
# Format: the paper's own numbering (arXiv:2609.30063v1).
# --------------------------------------------------------------------------
SECTIONS = {
    "abstract": "Abstract",
    "sec1": "§1 Introduction",
    "sec2": "§2 Self-Play Pretraining with Zero Data",
    "sec2.1": "§2.1 Program space",
    "sec2.2": "§2.2 Objectives",
    "sec3.1": "§3.1 Universal zero-shot transfer scaling laws",
    "sec3.2": "§3.2 In Context Learning",
    "sec4": "§4 Explaining Self-Play Scaling laws via Universal Data Ansatz",
    "sec5": "§5 Related Work",
    "sec6": "§6 Discussion",
    "appA.1": "§A.1 Pre-Pretraining with Self-Play Accelerates Pretraining on Natural Data",
    "appA.2": "§A.2 Pre-pretraining details",
    "appB": "§B Benchmark details",
    "appC": "§C Emergent Mathematical Structure",
    "appD": "§D ICL Tasks",
    "appE": "§E Brainfuck details",
    "appF": "§F Reward Ablations Table",
    "appG": "§G Pool construction",
    "appH": "§H Random-PCFG pretraining",
    "fig1": "Figure 1",
    "fig2": "Figure 2",
    "fig3": "Figure 3",
    "fig4": "Figure 4",
    "fig5": "Figure 5",
    "fig6": "Figure 6",
    "fig7": "Figure 7",
    "fig8": "Figure 8",
    "fig9": "Figure 9",
    "fig10": "Figure 10",
    "eq1": "Equation 1",
    "eq2": "Equation 2",
    "eq3": "Equation 3",
    "eq4": "Equation 4",
    "eq5": "Equation 5",
    "eq6": "Equation 6",
    "eq7": "Equation 7",
    "table1": "Table 1",
    "table2": "Table 2",
    "table3": "Table 3",
    "table4": "Table 4",
    "table5": "Table 5",
}

# --------------------------------------------------------------------------
# Model ladder. From the authors' released repo README (the folder labels are
# theirs; `paper_params` is the count the paper quotes).
# ------------------------------------------------------------------------#
RUNGS = [
    {"rung": "d64h1L1", "label": "100k", "paper_params": 99_000,
     "d_model": 64, "n_heads": 1, "n_layers": 1, "released_params": 98_496},
    {"rung": "d128h2L2", "label": "500k", "paper_params": 558_000,
     "d_model": 128, "n_heads": 2, "n_layers": 2, "released_params": 557_696},
    {"rung": "d128h2L4", "label": "1M", "paper_params": 1_000_000,
     "d_model": 128, "n_heads": 2, "n_layers": 4, "released_params": 1_049_728},
    {"rung": "d256h4L4", "label": "3M", "paper_params": 3_100_000,
     "d_model": 256, "n_heads": 4, "n_layers": 4, "released_params": 3_148_032},
    {"rung": "d256h4L8", "label": "6M", "paper_params": 6_200_000,
     "d_model": 256, "n_heads": 4, "n_layers": 8, "released_params": 6_164_736},
    {"rung": "d512h8L8", "label": "24M", "paper_params": 24_400_000,
     "d_model": 512, "n_heads": 8, "n_layers": 8, "released_params": 24_388_096},
]

# Tokens per self-play round, per checkpoint, per ensemble member. These are the
# constants in the authors' scaling_analysis.py and fix the compute axis:
#   C = K * N * POOL * CTX * (round + 1)
POOL = 1536
CTX = 4096

# --------------------------------------------------------------------------
# Pretraining arms compared in Figure 2.
# --------------------------------------------------------------------------
ARMS = {
    "selfplay": {
        "label": "Self-Play Pretraining with Zero Data",
        "short": "Self-play",
        "csv": "selfplay_frontier_perk.csv",
        "color": "#2a9d8f",
        "dash": None,          # solid line
        "blurb": "Learned program distribution, RL on learning progress (§2.2, Figure 2).",
    },
    "uniform": {
        "label": "Pretraining on programs sampled from a universal prior",
        "short": "Universal prior",
        "csv": "uniform_frontier_perk.csv",
        "color": "#e76f51",
        "dash": "6 4",
        "blurb": "Same program space, fixed Solomonoff-style prior (§3.1, Figure 2).",
    },
    "pcfg": {
        "label": "Pretrain on PCFG",
        "short": "PCFG",
        "csv": "pcfg_frontier_perk.csv",
        "color": "#7b6cd9",
        "dash": "2 3",
        "blurb": "Hand-designed probabilistic context-free grammars (§H, Figure 2).",
    },
}

# --------------------------------------------------------------------------
# Corpora. `paper_label` is the Table 2 row label; `group` is Table 2's grouping.
# `in_table2` marks the ten datasets make_latex_table.py actually renders.
# Binary corpora are excluded from the table by the authors, with a reason.
# --------------------------------------------------------------------------
CORPORA = [
    # key,                paper_label,                        group,                  in_table2
    ("dclm",                "text (dclm)",                    "Text",                 True),
    ("dclm_ranked",         "text (ranked vocab)",             "Text",                 False),
    ("kolmogorov_text",     "enwik9 text",                     "Text",                 False),
    ("llm_compression_cc",  "web text (CC)",                   "Text",                 False),
    ("cifar10_rgb_planar",  "CIFAR-10 image bytes",            "Images",               True),
    ("cifar10_rgb_hwc",     "CIFAR-10 (HWC interl.)",          "Images",               True),
    ("audio_8bit",          "audio 8-bit PCM",                 "Audio / speech",       True),
    ("audio_16bit",         "audio 16-bit PCM",                "Audio / speech",       True),
    ("esc50_pcm8",          "ESC-50 44.1 kHz PCM8",            "Audio / speech",       False),
    ("esc50_pcm8_11khz",    "ESC-50 11 kHz PCM8",              "Audio / speech",       False),
    ("musicnet_pcm8",       "MusicNet 44.1 kHz PCM8",          "Audio / speech",       False),
    ("musicnet_pcm8_11khz", "MusicNet 11 kHz PCM8",            "Audio / speech",       False),
    ("speech_commands_pcm8", "speech 16 kHz PCM8",            "Audio / speech",       False),
    ("speech_commands_pcm8_8khz", "speech 8 kHz PCM8",        "Audio / speech",       False),
    ("mutopia_melody_16th", "MIDI (Mutopia, 16th note grid)",  "Audio / speech",       True),
    ("metamath",            "Metamath set.mm",                 "Math / formal",        True),
    ("llm_compression_arxiv_math", "arXiv math",              "Math / formal",         False),
    ("dna",                 "DNA (8-symbol)",                  "Biological sequences", True),
    ("aitdcc_b_c_source",   "AITDCC C source",                 "Code",                 True),
    ("llm_compression_python", "Python source (GitHub)",       "Code",                 True),
    ("arithmetic",          "arithmetic",                      "excluded",             False),
    ("aitdcc_a_protein",    "protein",                         "excluded",             False),
    ("kolmogorov_dna",      "KoLMogorov DNA",                  "excluded",             False),
    ("aitdcc_d_glibc_rand", "glibc rand (binary)",             "excluded",             False),
    ("aitdcc_e_atlas_float32", "ATLAS float32 (binary)",      "excluded",             False),
    ("aitdcc_g_astronomy",  "astronomy (binary)",              "excluded",             False),
]

# --------------------------------------------------------------------------
# Literature exponents, from Table 2's caption and literature_references.md.
# `values` are the published compute exponents; `chinchilla` entries are
# (alpha, beta) pairs the authors convert via b = alpha*beta/(alpha+beta), which
# the app must compute rather than hardcode.
# --------------------------------------------------------------------------
LITERATURE = {
    "dclm": {"values": [0.048], "chinchilla": [(0.18, 0.22)],
             "refs": ["Henighan et al. 2020", "Aghajanyan et al. 2023"]},
    "cifar10_rgb_planar": {"values": [], "chinchilla": [(0.13, 0.13)],
                           "extra_values": [0.10],
                           "refs": ["Aghajanyan et al. 2023", "Henighan et al. 2020"]},
    "audio_8bit": {"values": [], "chinchilla": [(0.25, 0.24), (0.31, 0.24)],
                   "refs": ["Cuervo & Marxer 2024", "Aghajanyan et al. 2023"]},
    "mutopia_melody_16th": {"values": [], "chinchilla": [], "dash": True, "refs": []},
    "metamath": {"values": [0.17], "chinchilla": [], "refs": ["Henighan et al. 2020"]},
    "dna": {"values": [0.01, 0.06], "chinchilla": [], "refs": ["Shah et al. 2026 (dnaHNet)"]},
    "aitdcc_b_c_source": {"values": [], "chinchilla": [(0.37, 0.32)],
                          "refs": ["Aghajanyan et al. 2023"]},
}

# Reward-ablation arms (Table 5 / Appendix F).
REWARD_ARMS = [
    {"key": "none", "label": "None (canonical)", "included": True,
     "desc": "r = |<P ⊙ grad L, theta_past - theta_now>| with theta_past = theta_floor(e/2).",
     "ref": "appF"},
    {"key": "uniform", "label": "uniform", "included": True,
     "desc": "Generator removed entirely: programs drawn i.i.d. uniform over the alphabet.",
     "ref": "appF"},
    {"key": "signed", "label": "signed", "included": True,
     "desc": "Drops the absolute value around the inner product.", "ref": "appF"},
    {"key": "shuffle", "label": "shuffle", "included": True,
     "desc": "Permutes rewards across the program pool, breaking the program-reward correspondence.",
     "ref": "appF"},
    {"key": "last_step", "label": "last_step", "included": True,
     "desc": "One-step window, theta_past = theta_{e-1}, instead of the e/2 lookback.",
     "ref": "appF"},
    {"key": "loss_delta", "label": "loss_delta", "included": True,
     "desc": "Realized progress L(theta_pre) - L(theta_post) instead of the first-order score.",
     "ref": "appF"},
    {"key": "negate", "label": "negate", "included": True,
     "desc": "Flips the reward's sign.", "ref": "appF"},
]

# Table 5's numbers are NOT transcribed here. They are DERIVED at build time by
# tools/build_data.py, following the authors' own make_table.py: every arm read
# at the 1M rung, K=4, round 8191, from frontier_traj_reward.json for the five
# ablations, frontier_traj_perk.json for 'none', and uniform_frontier_perk.csv for
# 'uniform'. Running their script regenerates reward_arms.tex byte-identically, so
# the derivation is exact.
#
# Why not transcribe: extracting Table 5 from the PDF mangles its columns (rows
# come out with nine values against seven headers, with last_step/negate
# misaligned). A hand-transcribed table would have shipped wrong numbers wearing a
# paper citation. See docs/decisions.md D2.
TABLE5_ROW_ORDER = [
    "dclm", "metamath", "aitdcc_b_c_source", "dna", "arithmetic",
    "audio_8bit", "audio_16bit", "mutopia_melody_16th", "cifar10_rgb_planar",
    "aitdcc_d_glibc_rand",
]
TABLE5_ROW_LABELS = {
    "dclm": "text (dclm)", "metamath": "Metamath", "aitdcc_b_c_source": "C source",
    "dna": "DNA (8-symbol)", "arithmetic": "arithmetic", "audio_8bit": "audio 8-bit PCM",
    "audio_16bit": "audio 16-bit PCM", "mutopia_melody_16th": "melody (Mutopia)",
    "cifar10_rgb_planar": "CIFAR-10 (planar)",
    "aitdcc_d_glibc_rand": "random bytes",
}
TABLE5_COLUMNS = ["none", "uniform", "signed", "shuffle", "last_step",
                  "loss_delta", "negate"]

# Table 1/3: mathematical families the generator discovered, with the earliest
# self-play round and the expected first round under uniform sampling (Appendix C).
MATH_FAMILIES = [
    {"family": "arithmetic", "label": "Arithmetic (mod 256)",
     "selfplay_round": 0, "prior_expected_round": 105,
     "prior_p": 9.3e-6, "prior_hits": 1526,
     "program": "S+[.++]", "terms": "1,3,5,7,9,..."},
    {"family": "fibonacci", "label": "Fibonacci (mod 256)",
     "selfplay_round": 512, "prior_expected_round": 53_000,
     "prior_p": None, "prior_hits": 0,
     "program": "S,[[.C>.C>]", "terms": "1,1,2,3,5,..."},
    {"family": "geometric", "label": "Geometric (mod 256)",
     "selfplay_round": 256, "prior_expected_round": 53_000,
     "prior_p": None, "prior_hits": 0,
     "program": "S+[.L>]", "terms": "1,3,9,27,81,..."},
    {"family": "quadratic", "label": "Quadratic (mod 256)",
     "selfplay_round": 512, "prior_expected_round": 53_000,
     "prior_p": None, "prior_hits": 0,
     "program": "S,.[<C>>VX<RX++]", "terms": "9,25,59,111,..."},
    {"family": "cubic", "label": "Cubic (mod 256)",
     "selfplay_round": 512, "prior_expected_round": 53_000,
     "prior_p": None, "prior_hits": 0,
     "program": "S+[[-.L>L>-]-]", "terms": "0,254,236,74,..."},
]
# Programs in the table above are the paper's illustrative examples; the earliest
# round values are recomputed from hits.jsonl and must agree (see verify_data.py).
MATH_PRIOR_TOTAL_SAMPLES = 164_000_000
MATH_PRIOR_PROGRAMS_PER_ROUND = 1024
MATH_PRIOR_RULE_OF_THREE = 1.8e-8

# ICL tasks (Appendix D). `blurb` is what the task demonstrates, per §3.2.
ICL_TASKS = {
    "max": {"label": "max", "blurb": "argmax over k input bytes", "ref": "appD"},
    "min": {"label": "min", "blurb": "argmin over k input bytes", "ref": "appD"},
    "sum": {"label": "sum", "blurb": "x1 + x2 mod 256", "ref": "appD"},
    "mean": {"label": "mean", "blurb": "mean of k input bytes", "ref": "appD"},
    "first": {"label": "first", "blurb": "first input byte", "ref": "appD"},
    "last": {"label": "last", "blurb": "last input byte", "ref": "appD"},
}
ICL_EXTRA_TASKS = {
    "assoc": {"label": "associative recall",
              "blurb": "Print a dictionary of byte pairs, then look up unseen keys. "
                       "Tests contextual search (§3.2).",
              "ref": "appD"},
    "index": {"label": "reverse string",
              "blurb": "Reverse the input byte sequence. Tests dynamic indexing (§3.2).",
              "ref": "appD"},
    "stack": {"label": "stack",
              "blurb": "Apply stack ops, then report the final pop. Tests simulating a "
                       "context-free grammar (§3.2).",
              "ref": "appD"},
    "succ": {"label": "successor", "blurb": "f(x) = x + 1", "ref": "appD"},
    "palin": {"label": "palindrome", "blurb": "Is the input a palindrome?", "ref": "appD"},
}

# The augmented Brainfuck alphabet (Appendix E, Table 4). Macro expansions are
# the authors', verbatim.
BF_MACROS = [
    {"token": "Z", "expansion": "[-]", "effect": "Clear current cell"},
    {"token": "R", "expansion": "[->+<]", "effect": "Clear and add x into right neighbor"},
    {"token": "L", "expansion": "[->+++<]", "effect": "Clear and add 3*x into right neighbor"},
    {"token": "N", "expansion": "[-<->]", "effect": "Clear and subtract x from left neighbor"},
    {"token": "C", "expansion": "[->+>+<<]", "effect": "Clear and add x into the two right cells"},
    {"token": "G", "expansion": "[>]", "effect": "Scan right to next zero cell"},
    {"token": "H", "expansion": "[<]", "effect": "Scan left to next zero cell"},
    {"token": "W", "expansion": "[[-]>+<]", "effect": "If current != 0: increment right and clear current"},
    {"token": "V", "expansion": "[.>]", "effect": "Print stored string until a zero cell"},
    {"token": "X", "expansion": "[-]", "effect": "Set cell to 16"},
]
BF_PRIMITIVES = [
    {"token": ">", "effect": "move head right"},
    {"token": "<", "effect": "move head left"},
    {"token": "+", "effect": "increment cell under head"},
    {"token": "-", "effect": "decrement cell under head"},
    {"token": "[", "effect": "loop start"},
    {"token": "]", "effect": "loop end"},
    {"token": ",", "effect": "read a uniform random byte from the input tape"},
    {"token": ".", "effect": "emit the cell under head"},
]
BF_PARAMS = {"cell_modulus": 256, "program_alphabet": "<>+-[].,F",
             "prefix_program": "S", "prefix_output": "O"}

# Our own CPU measurements, taken on this box, as a live cross-check against
# the paper. Filled by tools/build_data.py from tools/measure_local.py output.
LOCAL_MEASUREMENT_NOTE = (
    "Measured locally on CPU against the authors' released 100k checkpoint "
    "(98,496 params, d64h1L1). Round 0 is the random initialization and scores "
    "worse than the uniform-256 baseline of 8.0 bits/byte, which is the expected "
    "control; trained rounds sit near 6.15."
)