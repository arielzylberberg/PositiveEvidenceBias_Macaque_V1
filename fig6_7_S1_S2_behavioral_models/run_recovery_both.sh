#!/bin/bash
cd /tmp/claude-0/-home-claude/b4d5361e-df4a-59c6-b001-5a8876f39df7/scratchpad/code_for_sharing2/fig6_7_S1_S2_behavioral_models
echo "=== Starting Ziggy $(date) ===" >> recovery_run.log
python3 model_recovery.py --animal Ziggy --n_replicates 20 --workers 2 >> recovery_run.log 2>&1
echo "=== Ziggy done, starting Friedrich $(date) ===" >> recovery_run.log
python3 model_recovery.py --animal Friedrich --n_replicates 20 --workers 2 >> recovery_run.log 2>&1
echo "=== ALL DONE $(date) ===" >> recovery_run.log
