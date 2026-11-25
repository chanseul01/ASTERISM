# Data license

ASTERISM keeps the license of each source benchmark.

- For sources whose license allows redistribution and commercial use, we release the items with their images, together with our corrected labels and answer formats, the rewritten items and the per-item labels.
- For sources with non-commercial terms or without a stated license, we release the item identifiers, our corrections and rewritten questions and options, and build scripts that assemble the items from the public release of each source.
- Where the dataset card and the repository or paper of a source state different licenses, we follow the more restrictive one.

Our annotations (corrected labels and answer formats, rewritten items and per-item labels) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), except in the 96 items whose `license` is CC BY-NC-SA 4.0 (MuirBench items from IconQA and ISVQA), which carry that license. Images and other source content keep the license of their source.

Some items carry a license other than that of their source, or further terms of their source or of the dataset their images come from. Each record gives these in `license`, `image_origin` and `use_terms`, and the ASTERISM dataset card lists them.

## ASTERISM sources

| Source | License | Release | Main | Retrieval | Rewrites | Items |
|---|---|---|---:|---:|---:|---:|
| MileBench | CC BY-NC 4.0 <sup>a</sup> | identifiers | 1,508 | 318 | 218 | 2,044 |
| MMRB | CC BY 4.0 | items | 1,069 | 191 | – | 1,260 |
| VLM2-Bench | CC BY-NC 4.0 <sup>a</sup> | identifiers | 1,728 | – | 82 | 1,810 |
| MMR-Life | CC BY 4.0 | items | 2,093 | – | 158 | 2,251 |
| ReMI | CC BY 4.0 <sup>b</sup> | items | 1,956 | – | 273 | 2,229 |
| MuirBench | CC BY 4.0 | items | 979 | 226 | – | 1,205 |
| MRAG-Bench | CC BY 4.0 | items | – | 291 | – | 291 |
| M4Bench | not stated | identifiers | 865 | – | – | 865 |
| MV-MATH | MIT | items | 857 | – | 36 | 893 |
| BLINK | Apache 2.0 | items | 879 | – | 145 | 1,024 |
| MMLongBench-Doc | CC BY-NC 4.0 | identifiers | – | 963 | – | 963 |
| DUDE | CC BY 4.0 | items | – | 334 | – | 334 |
| MaRVL | CC BY 4.0 (text), CC BY-NC-SA 4.0 (images) | identifiers | 316 | – | 147 | 463 |
| MMSI-Bench | CC BY 4.0 | items | 928 | – | 23 | 951 |
| Q-Bench2 | S-Lab License 1.0 (non-commercial) <sup>a</sup> | identifiers | 391 | – | 102 | 493 |
| MIRB | MIT | items | 432 | 4 | 177 | 613 |
| DiCoBench | MIT | items | 547 | – | 22 | 569 |
| EMMA | not stated | identifiers | 309 | – | – | 309 |
| Mantis-Eval | Apache 2.0 | items | 82 | 24 | 23 | 129 |
| Total | | | 14,939 | 2,351 | 1,406 | 18,696 |

Main counts the source items in the six task categories other than retrieval, Retrieval the source items in the retrieval category, and Rewrites the rewritten items derived from the source.

<sup>a</sup> The dataset card and the repository or paper state different licenses, and the more restrictive one is listed.<br>
<sup>b</sup> Named on the dataset card, whose license file is empty.

Images keep the licenses of the datasets they come from.

## ASTERISM-Train sources

ASTERISM-Train follows the same rule for its 48 sources. The license and the release mode of each source are listed on the ASTERISM-Train dataset card. Items from M4-Instruct, including the WebQA items taken from it, are also subject to the terms on the M4-Instruct dataset card, and items from RAVEN keep the GPL-3.0 of RAVEN. Our annotations, including the synthetic items, are released under CC BY 4.0, except in the 2,499 items whose `license` is CC BY-NC-SA 4.0 (IconQA and nuScenes), which carry that license.

## Intended use

ASTERISM and ASTERISM-Lite are for evaluating MLLMs, and ASTERISM-Train is the set intended for training. Training on ASTERISM items, identifying or profiling the people shown in its images, and commercial use of items whose sources carry non-commercial terms are out of scope.
