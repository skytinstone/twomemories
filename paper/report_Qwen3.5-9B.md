# 단계 보고 / Phase report: all (Qwen/Qwen3.5-9B)

## 컴퓨팅 / Compute

| GPU | cells | samples | sec | sec/sample | units (est.) | peak VRAM GB |
|---|---|---|---|---|---|---|
| NVIDIA A100-SXM4-80GB | 163 | 16294 | 23531 | 1.44 | 58.04 | 9.3 |

## Table 2: 충실도 × 태스크, retained [95% CI] / fidelity x task

| condition | DocVQA | ChartQA | TextVQA | POPE |
|---|---|---|---|---|
| KV 4 / State full | n/e | n/e | n/e | n/e |
| KV noise / State full | n/e | n/e | n/e | n/e |
| KV zero / State full | n/e | n/e | n/e | n/e |
| KV full / State 8 | n/e | n/e | n/e | n/e |
| KV full / State 8ef | n/e | n/e | n/e | n/e |
| KV full / State 6 | n/e | n/e | n/e | n/e |
| KV full / State 6ef | n/e | n/e | n/e | n/e |
| KV full / State noise | n/e | n/e | n/e | n/e |
| KV full / State zero | n/e | n/e | n/e | n/e |
| KV zero / State zero | n/e | n/e | n/e | n/e |
| KV 4 / State 8 | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef | n/e | n/e | n/e | n/e |
| KV 4 / State 8 (whole) | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef (whole) | n/e | n/e | n/e | n/e |

## Table 5: 토큰 × KV 비트 / tokens x KV bits (같은 MB 묶음 / iso-MB groups)

| task | tokens | KV bits | total MB | retained [95% CI] | MB group |
|---|---|---|---|---|---|
| DocVQA | 100% | full | n/e | n/e | - |
| DocVQA | 100% | 4 | n/e | n/e | - |
| DocVQA | 50% | full | n/e | n/e | - |
| DocVQA | 50% | 4 | n/e | n/e | - |
| DocVQA | 25% | full | n/e | n/e | - |
| DocVQA | 25% | 4 | n/e | n/e | - |
| ChartQA | 100% | full | n/e | n/e | - |
| ChartQA | 100% | 4 | n/e | n/e | - |
| ChartQA | 50% | full | n/e | n/e | - |
| ChartQA | 50% | 4 | n/e | n/e | - |
| ChartQA | 25% | full | n/e | n/e | - |
| ChartQA | 25% | 4 | n/e | n/e | - |
| TextVQA | 100% | full | n/e | n/e | - |
| TextVQA | 100% | 4 | n/e | n/e | - |
| TextVQA | 50% | full | n/e | n/e | - |
| TextVQA | 50% | 4 | n/e | n/e | - |
| TextVQA | 25% | full | n/e | n/e | - |
| TextVQA | 25% | 4 | n/e | n/e | - |
| POPE | 100% | full | n/e | n/e | - |
| POPE | 100% | 4 | n/e | n/e | - |
| POPE | 50% | full | n/e | n/e | - |
| POPE | 50% | 4 | n/e | n/e | - |
| POPE | 25% | full | n/e | n/e | - |
| POPE | 25% | 4 | n/e | n/e | - |

n/e = 평가 안 함 / not evaluated. retained = (NLL_blind − NLL_cond)/(NLL_blind − NLL_full), 데이터셋 수준 / dataset level, paired bootstrap n=2000.

