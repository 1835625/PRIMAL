文件说明

1. primal_testing_obs10.py
   适配 thesis-improve 分支 10 通道 ACNet + mapf_gym.py 的测试脚本。
   不再使用原来的 mapf_gym_cap.py。

2. summarize_primal_results.py
   用来汇总一个或两个结果目录，输出整体、按 size、按 agent 数、按 density 的对比表。

Windows 本地示例

1) 运行 10 通道模型测试
python primal_testing_obs10.py ^
  --model-dir model_obs10 ^
  --env-dir saved_environments ^
  --results-dir primal_results_obs10 ^
  --save-solution ^
  --cpu-only

2) 汇总单个模型结果
python summarize_primal_results.py ^
  --results-a primal_results_obs10 ^
  --label-a obs10 ^
  --out-dir compare_summary_obs10

3) 汇总两个模型结果
python summarize_primal_results.py ^
  --results-a primal_results_baseline ^
  --label-a baseline ^
  --results-b primal_results_obs10 ^
  --label-b obs10 ^
  --out-dir compare_summary_baseline_vs_obs10
