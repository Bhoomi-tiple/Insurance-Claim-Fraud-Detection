"""Run the whole pipeline in order:  python run_all.py
(The test set is only touched in s6_test_eval - run the full pipeline once you are happy with the setup.)"""
import time

from src import (s1_eda, s2_split, s3_baselines, s4_tuning, s5_ablation,
                 s6_test_eval, s7_explain, s8_export_app_assets)

STAGES = [("Stage 2  split", s2_split), ("Stage 1  EDA (train only)", s1_eda),
          ("Stage 3-4 baselines + CV", s3_baselines), ("Stage 5  tuning + nested CV", s4_tuning),
          ("Stage 6  ablation", s5_ablation), ("Stage 7  test evaluation", s6_test_eval),
          ("Stage 8  explainability", s7_explain), ("Stage 8  app assets", s8_export_app_assets)]

if __name__ == "__main__":
    for title, module in STAGES:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        t = time.time()
        module.main()
        print(f"[done in {time.time() - t:.0f}s]")
