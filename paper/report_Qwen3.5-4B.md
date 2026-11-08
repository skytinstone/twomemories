# 단계 보고 / Phase report: all (Qwen/Qwen3.5-4B)

## 컴퓨팅 / Compute

| GPU | cells | samples | sec | sec/sample | units (est.) | peak VRAM GB |
|---|---|---|---|---|---|---|
| NVIDIA A100-SXM4-40GB | 56 | 2800 | 1626 | 0.58 | 4.01 | 9.3 |

## Table 2: 충실도 × 태스크, retained [95% CI] / fidelity x task

| condition | DocVQA | ChartQA | TextVQA | POPE |
|---|---|---|---|---|
| KV 4 / State full | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| KV noise / State full | n/e | n/e | n/e | n/e |
| KV zero / State full | 0.33 [0.29, 0.37] | 0.41 [0.36, 0.47] | 0.51 [0.47, 0.55] | 0.96 [0.93, 0.99] |
| KV full / State 8 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| KV full / State 8ef | n/e | n/e | n/e | n/e |
| KV full / State 6 | n/e | n/e | n/e | n/e |
| KV full / State 6ef | n/e | n/e | n/e | n/e |
| KV full / State noise | n/e | n/e | n/e | n/e |
| KV full / State zero | 1.00 [0.99, 1.02] | 1.01 [0.99, 1.04] | 1.00 [0.99, 1.00] | 0.58 [0.50, 0.65] |
| KV zero / State zero | n/e | n/e | n/e | n/e |
| KV 4 / State 8 | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef | 1.00 [0.99, 1.00] | 1.00 [0.99, 1.01] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| KV 4 / State 8 (whole) | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef (whole) | n/e | n/e | n/e | n/e |
| KV drop / State full | n/e | n/e | n/e | n/e |
| KV full / State zeroconv | n/e | n/e | n/e | n/e |
| KV drop / State zeroconv | n/e | n/e | n/e | n/e |

## Table 5: 토큰 × KV 비트 / tokens x KV bits (같은 MB 묶음 / iso-MB groups)

| task | tokens | KV bits | total MB | retained [95% CI] | MB group |
|---|---|---|---|---|---|
| DocVQA | 100% | full | 91.13 | 1.00 [1.00, 1.00] | A |
| DocVQA | 100% | 4 | 60.53 | 1.00 [0.99, 1.00] | B |
| DocVQA | 50% | full | n/e | n/e | - |
| DocVQA | 50% | 4 | n/e | n/e | - |
| DocVQA | 25% | full | n/e | n/e | - |
| DocVQA | 25% | 4 | n/e | n/e | - |
| ChartQA | 100% | full | 64.46 | 1.00 [1.00, 1.00] | A |
| ChartQA | 100% | 4 | 53.86 | 1.00 [0.99, 1.01] | B |
| ChartQA | 50% | full | n/e | n/e | - |
| ChartQA | 50% | 4 | n/e | n/e | - |
| ChartQA | 25% | full | n/e | n/e | - |
| ChartQA | 25% | 4 | n/e | n/e | - |
| TextVQA | 100% | full | 75.39 | 1.00 [1.00, 1.00] | A |
| TextVQA | 100% | 4 | 56.60 | 1.00 [1.00, 1.00] | B |
| TextVQA | 50% | full | n/e | n/e | - |
| TextVQA | 50% | 4 | n/e | n/e | - |
| TextVQA | 25% | full | n/e | n/e | - |
| TextVQA | 25% | 4 | n/e | n/e | - |
| POPE | 100% | full | 60.10 | 1.00 [1.00, 1.00] | A |
| POPE | 100% | 4 | 52.77 | 1.00 [1.00, 1.00] | B |
| POPE | 50% | full | n/e | n/e | - |
| POPE | 50% | 4 | n/e | n/e | - |
| POPE | 25% | full | n/e | n/e | - |
| POPE | 25% | 4 | n/e | n/e | - |

n/e = 평가 안 함 / not evaluated. retained = (NLL_blind − NLL_cond)/(NLL_blind − NLL_full), 데이터셋 수준 / dataset level, paired bootstrap n=2000.

