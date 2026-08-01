# dreamzero-yam

LoRA post-train of NVIDIA [DreamZero](https://github.com/dreamzero0/dreamzero) (14B world-action model, base `GEAR-Dreams/DreamZero-AgiBot`) on the 124-repo MolmoAct2 BimanualYAM split, trained on Modal — the first public DreamZero fine-tune on this dataset.

Target: `robocurve/dreamzero-yam-molmoact2` on HF, with held-out open-loop MSE on `allenai/19012026-block-13` compared against our [π0.5](https://huggingface.co/robocurve/pi05-yam-molmoact2) (0.00206) and [GR00T N1.7](https://huggingface.co/robocurve/gr00t-n1.7-yam-molmoact2) (0.00279) replications under a unified re-eval protocol.

**Plan:** [plans/0001-dreamzero-yam-finetune.md](plans/0001-dreamzero-yam-finetune.md) (6 critique rounds, approved 2026-07-31; budget cap $5k)

Status: Phase 0 (recon) — `gr00t-yam-data` volume verified to still hold the converted LeRobot-v2 + H.264 tree.
