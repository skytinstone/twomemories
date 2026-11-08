**table2** model=Qwen/Qwen3.5-4B

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
