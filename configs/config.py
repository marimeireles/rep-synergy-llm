CONFIG = {
    "model_name": "EleutherAI/pythia-1b",
    "num_layers": 16,
    "num_heads_per_layer": 16,
    "num_total_heads": 256,  # 16 * 16
    "num_tokens_to_generate": 100,
    "seed": 42,
    "device": "auto",
    "results_dir": "results",
    # Generation: greedy decoding (deterministic, reproducible)
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    # PhiID
    "phiid_tau": 1,
    "phiid_kind": "gaussian",
    "phiid_redundancy": "MMI",
    # Ablation
    "num_random_ablation_seeds": 5,
}
