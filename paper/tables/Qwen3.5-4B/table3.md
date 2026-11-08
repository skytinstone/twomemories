**table3** model=Qwen/Qwen3.5-4B

| condition | DocVQA | ChartQA | TextVQA | POPE |
|---|---|---|---|---|
| unaltered | 0.923 | 0.800 | 0.795 | 0.880 / 0.854 |
| blind (no image) | 0.060 | 0.110 | 0.062 | 0.590 / 0.047 |
| KV 4 / State full | 0.923 | 0.800 | 0.786 | 0.880 / 0.854 |
| KV noise / State full | n/e | n/e | n/e | n/e |
| KV zero / State full | 0.103 | 0.150 | 0.105 | 0.710 / 0.491 |
| KV full / State 8 | 0.923 | 0.800 | 0.795 | 0.880 / 0.854 |
| KV full / State 8ef | n/e | n/e | n/e | n/e |
| KV full / State 6 | n/e | n/e | n/e | n/e |
| KV full / State 6ef | n/e | n/e | n/e | n/e |
| KV full / State noise | n/e | n/e | n/e | n/e |
| KV full / State zero | 0.953 | 0.800 | 0.813 | 0.900 / 0.875 |
| KV zero / State zero | n/e | n/e | n/e | n/e |
| KV 4 / State 8 | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef | 0.923 | 0.800 | 0.799 | 0.880 / 0.854 |
| KV 4 / State 6 | n/e | n/e | n/e | n/e |
| KV 4 / State 6ef | n/e | n/e | n/e | n/e |
| KV 4 / State 8 (whole) | n/e | n/e | n/e | n/e |
| KV 4 / State 8ef (whole) | n/e | n/e | n/e | n/e |
| KV drop / State full | n/e | n/e | n/e | n/e |
| KV full / State zeroconv | n/e | n/e | n/e | n/e |
| KV drop / State zeroconv | n/e | n/e | n/e | n/e |
