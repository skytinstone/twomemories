> mps/cpu rows included (results=results/mac); paper numbers only from results/mac (D-30).

**table3** model=Qwen/Qwen3.5-0.8B

| condition | DocVQA | ChartQA | TextVQA | POPE |
|---|---|---|---|---|
| unaltered | 0.880 | 0.745 | 0.726 | 0.867 / 0.856 |
| blind (no image) | 0.053 | 0.100 | 0.062 | 0.463 / 0.462 |
| KV 4 / State full | 0.877 | 0.745 | 0.720 | 0.860 / 0.850 |
| KV noise / State full | 0.877 | 0.745 | 0.725 | 0.870 / 0.860 |
| KV zero / State full | 0.066 | 0.115 | 0.107 | 0.677 / 0.553 |
| KV full / State 8 | 0.883 | 0.745 | 0.718 | 0.867 / 0.856 |
| KV full / State 8ef | 0.878 | 0.740 | 0.718 | 0.863 / 0.852 |
| KV full / State 6 | 0.874 | 0.730 | 0.727 | 0.863 / 0.854 |
| KV full / State 6ef | 0.870 | 0.740 | 0.725 | 0.867 / 0.857 |
| KV full / State noise | 0.855 | 0.720 | 0.715 | 0.863 / 0.851 |
| KV full / State zero | 0.857 | 0.525 | 0.678 | 0.830 / 0.805 |
| KV zero / State zero | 0.031 | 0.100 | 0.061 | 0.583 / 0.309 |
| KV 4 / State 8 | 0.877 | 0.740 | 0.717 | 0.863 / 0.854 |
| KV 4 / State 8ef | 0.859 | 0.740 | 0.718 | 0.863 / 0.853 |
| KV 4 / State 6 | 0.877 | 0.720 | 0.723 | 0.863 / 0.854 |
| KV 4 / State 6ef | 0.854 | 0.740 | 0.732 | 0.867 / 0.857 |
| KV 4 / State 8 (whole) | 0.872 | 0.720 | 0.710 | 0.860 / 0.846 |
| KV 4 / State 8ef (whole) | 0.868 | 0.725 | 0.717 | 0.860 / 0.846 |
| KV drop / State full | 0.072 | 0.120 | 0.106 | 0.680 / 0.560 |
| KV full / State zeroconv | 0.623 | 0.310 | 0.521 | 0.657 / 0.713 |
| KV drop / State zeroconv | 0.048 | 0.095 | 0.053 | 0.457 / 0.407 |
