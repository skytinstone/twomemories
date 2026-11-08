**table2** model=Qwen/Qwen3.5-4B

| condition | DocVQA | ChartQA | TextVQA | POPE |
|---|---|---|---|---|
| KV 4 / State full | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.99 [0.97, 1.00] | 1.00 [1.00, 1.00] |
| KV noise / State full | n/e | n/e | n/e | n/e |
| KV zero / State full | 0.05 [-0.01, 0.11] | 0.06 [-0.04, 0.14] | 0.06 [-0.02, 0.14] | 0.41 [0.21, 0.67] |
| KV full / State 8 | 1.00 [1.00, 1.00] | 1.00 [0.96, 1.04] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| KV full / State 8ef | n/e | n/e | n/e | n/e |
| KV full / State 6 | n/e | n/e | n/e | n/e |
| KV full / State 6ef | n/e | n/e | n/e | n/e |
| KV full / State noise | n/e | n/e | n/e | n/e |
| KV full / State zero | 1.04 [1.00, 1.08] | 1.00 [0.94, 1.06] | 1.02 [0.97, 1.08] | 1.07 [0.95, 1.27] |
| KV zero / State zero | n/e | n/e | n/e | n/e |
| KV 4 / State 8 | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.01 [0.99, 1.03] | 1.00 [1.00, 1.00] |
| KV 4 / State 6 | n/e | n/e | n/e | n/e |
| KV 4 / State 6ef | n/e | n/e | n/e | n/e |
| KV 4 / State 8 (whole) | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef (whole) | n/e | n/e | n/e | n/e |
| KV drop / State full | n/e | n/e | n/e | n/e |
| KV full / State zeroconv | n/e | n/e | n/e | n/e |
| KV drop / State zeroconv | n/e | n/e | n/e | n/e |
