# 01_split_seed

Varia gerador e seed do split; Random Forest fixo em 42.

```json
{
  "generators": [
    "pcg64_cp_alite1",
    "numpy_pcg64",
    "numpy_mt19937"
  ],
  "split_seeds": [
    1000,
    1001,
    1002,
    1003,
    1004,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1024,
    1025,
    1026,
    1027,
    1028,
    1029
  ],
  "model_random_state": 42,
  "dataset": "data/raw/galaxies.csv"
}
```
