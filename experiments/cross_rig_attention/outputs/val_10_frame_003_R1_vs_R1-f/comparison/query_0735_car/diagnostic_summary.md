# Paired diagnostic summary: query_0735_car

## Final-layer outcomes

| Condition | Query | GT mass | Entropy | Confidence | BEV error (m) |
|---|---:|---:|---:|---:|---:|
| R1 object-matched | 735 | 0.0246 | 0.6773 | 0.9819 | 0.277 |
| R1-f object-matched | 15 | 0.0171 | 0.5415 | 0.8953 | 4.774 |
| R1-f fixed R1 query | 735 | 0.0109 | 0.5615 | 0.8529 | 3.549 |

## First-layer term diagnosis

| Term | R1 object-matched discrimination / utility | R1-f object-matched discrimination / utility | R1-f fixed R1 query discrimination / utility |
|---|---:|---:|---:|
| H-X | 0.354 / -0.0027 | 0.150 / -0.0019 | 0.150 / -0.0008 |
| H-G3D | 0.182 / -0.0225 | -0.102 / +0.0059 | -0.102 / -0.0037 |
| H-GMV | 1.163 / +0.0310 | 1.105 / +0.0213 | 1.105 / +0.0035 |
| E-X | 0.114 / +0.0067 | 0.194 / +0.0018 | 0.136 / +0.0008 |
| E-G3D | 0.243 / +0.0222 | 0.402 / +0.0170 | 0.223 / -0.0122 |
| E-GMV | -0.221 / +0.0117 | 0.080 / -0.0008 | -0.121 / -0.0035 |

Positive discrimination means the term scores GT-consistent rays above other valid rays. Positive utility means removing the term reduces GT attention mass; negative utility means the term is currently directing attention away from the GT.
