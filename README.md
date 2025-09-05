# Beyond Speech: Exploring the Effects of Visual Grounding on Speech Representations

## 0. Clone the Repo and Install Environments
Please clone this repository and install the required libraries listed in the following files:
- `requirements_analysis.txt`
- `requirements_representation_extraction.txt`

## 1. Model Weights
Model weights for **wav2vec 2.0-base** and **BERT-base** can be accessed via HuggingFace:
- [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- [BERT-base](https://huggingface.co/google-bert/bert-base-uncased)

**FaST-VGS+** and **VG-BERT** weights can be downloaded from the following sources:
- [FaST-VGS+ weights](https://drive.google.com/drive/folders/1AOSXSaEgP8vnBR3cjLI7k_IYsFk_uZD3)
- [VG-BERT weights](https://drive.google.com/file/d/1icYBK4MJ7KYWWkoMeJoXFALYBtIn3Yvo/view?usp=sharing)

These links were taken from the official repositories:
- [FaST-VGS-Family GitHub](https://github.com/jasonppy/FaST-VGS-Family)
- [VG-BERT GitHub](https://github.com/yizhen-zhang/VG-Bert)

## 2. Download Data
Our newly created datasets of semantically similar and phonetically different words (or vice versa) can be found in `semantic_categories.json` and `phonetic_groups.json`, respectively.  
Below, you can find the publicly available datasets that we used in our research:
- **MALD Data**: [MALD database](http://mald.artsrn.ualberta.ca/) (Item data and "Words" from Audio Files)
- **Librispeech**: [Librispeech dev-clean and test-clean](https://www.openslr.org/12)
- **Librispeech Alignments**: [Alignments from Zenodo](https://zenodo.org/records/2619474#.XKDP2VNKg1g)
- **Concreteness Ratings**: [Concreteness ratings Excel file](https://link.springer.com/article/10.3758/s13428-013-0403-5#MOESM1)
- **GloVe Embeddings**: [glove.6B.zip](https://nlp.stanford.edu/projects/glove/)

## 3. Extract Embeddings
To extract embeddings from the data:
- **MALD**: Check the arguments in `get_fast_vgs_embeddings.py` and `get_w2v2_embeddings.py`, then run these files.
- **LibriSpeech**: Run `dataset_cleanup.py` (following Choi et al., 2024), then proceed as for MALD.
  - `args.slice = True` performs audio slicing.
  - `args.slice = False` performs feature slicing.

## 4. Analysis
Follow the steps described in the respective notebooks (`*.ipynb`) for analysis.

## 5. Datasets

### Semantic Categories

### Semantic categories with words, average concreteness ratings, average cosine similarities of GloVe embeddings, and average pairwise phonetic (Levenshtein) distances.  
_All ± values indicate one standard deviation._

---

## **Concrete Categories**

| Category            | Words | Avg. Concr. | Avg. Cos. Sim. | Avg. Phon. Dist. |
|---------------------|-------|-------------|----------------|------------------|
| **Musical Instruments** | piano, guitar, violin, trumpet, flute, saxophone, cello, drum, clarinet, harp | 4.91 ± 0.08 | 0.63 ± 0.09 | 0.64 ± 0.11 |
| **Clothing Articles**   | shirt, pants, jacket, sweater, coat, dress, skirt, socks, gloves, hat, shoes, boots, scarf, jeans, blouse, hoodie, belt, shorts, tie | 4.87 ± 0.12 | 0.48 ± 0.11 | 0.63 ± 0.10 |
| **Fruits**              | apple, banana, orange, mango, pear, grape, peach, lemon, cherry, kiwi, plum, watermelon, melon, fig, papaya, coconut, apricot, nectarine, raspberry | 4.86 ± 0.16 | 0.52 ± 0.09 | 0.64 ± 0.13 |
| **Vehicles**            | car, bus, truck, train, van, bicycle, airplane, scooter, boat, taxi, tram, subway, helicopter, motorbike | 4.85 ± 0.09 | 0.45 ± 0.11 | 0.68 ± 0.12 |
| **Building Materials**  | brick, concrete, wood, cement, steel, stone, gravel, glass, tile, sand, asphalt, plaster, insulation, drywall, lumber, clay | 4.78 ± 0.14 | 0.41 ± 0.11 | 0.67 ± 0.12 |
| **Organs**              | heart, liver, brain, kidney, lung, stomach, pancreas, bladder | 4.65 ± 0.13 | 0.49 ± 0.10 | 0.65 ± 0.11 |

---

## **Abstract Categories**

| Category            | Words | Avg. Concr. | Avg. Cos. Sim. | Avg. Phon. Dist. |
|---------------------|-------|-------------|----------------|------------------|
| **Financial Terms** | equity, asset, liability, revenue, expense, debt, profit, dividend, investment, capital, tax, budget, interest | 2.11 ± 0.40 | 0.42 ± 0.10 | 0.62 ± 0.10 |
| **Emotions**        | joy, sadness, anger, fear, love, pride, shame, envy, anxiety, disgust | 2.10 ± 0.41 | 0.43 ± 0.10 | 0.70 ± 0.14 |
| **Ethical/Legal Terms** | justice, fairness, honesty, integrity, law, duty, rights, punishment | 1.84 ± 0.36 | 0.38 ± 0.10 | 0.64 ± 0.11 |

### Phonetic Groups
### Phonetic groups with words, average concreteness ratings, average cosine similarities of GloVe embeddings, and average pairwise phonetic (Levenshtein) distances.  
_All ± values indicate one standard deviation._

---

## **Concrete Categories**

| Group       | Words | Avg. Concreteness | Avg. Cos. Sim. | Avg. Phon. Dist. |
|-------------|-------|-------------------|----------------|------------------|
| **Concrete 1** | liquor, kicker, ticker, litter, scissor, mirror, simmer, fissure, shipper | 4.28 ± 0.46 | 0.01 ± 0.06 | 0.20 ± 0.06 |
| **Concrete 2** | linebacker, rainwater, firefighter, whitewater, gunpowder, songwriter, contact, roadrunner | 4.53 ± 0.30 | 0.00 ± 0.05 | 0.33 ± 0.09 |
| **Concrete 3** | chin, chip, chuck, chum, chug, hymn | 4.16 ± 0.47 | -0.01 ± 0.06 | 0.23 ± 0.08 |
| **Concrete 4** | handshake, handbrake, handbook, handmaid, handmade | 4.42 ± 0.26 | 0.04 ± 0.04 | 0.20 ± 0.08 |
| **Concrete 5** | grin, grid, brig, groin, brook | 4.44 ± 0.15 | 0.00 ± 0.05 | 0.29 ± 0.10 |
| **Concrete 6** | hothouse, chophouse, courthouse, harness, houseboat | 4.49 ± 0.30 | 0.00 ± 0.06 | 0.36 ± 0.12 |
| **Concrete 7** | lap, map, cap, lamb, latch, lawn, pal, dam | 4.64 ± 0.38 | 0.03 ± 0.06 | 0.29 ± 0.09 |

---

## **Abstract Categories**

| Group       | Words | Avg. Concreteness | Avg. Cos. Sim. | Avg. Phon. Dist. |
|-------------|-------|-------------------|----------------|------------------|
| **Abstract 1** | coyness, heinous, famous, cautious, focal | 1.94 ± 0.15 | 0.00 ± 0.06 | 0.32 ± 0.08 |
| **Abstract 2** | mostly, ghostly, grossly, manly, keenly | 1.93 ± 0.23 | 0.01 ± 0.04 | 0.30 ± 0.08 |
| **Abstract 3** | deduction, defensive, detraction, affection, dominion | 2.08 ± 0.11 | 0.00 ± 0.04 | 0.30 ± 0.06 |
| **Abstract 4** | inherently, incoherently, indefinitely, inadvertently, heavenly | 1.73 ± 0.32 | 0.04 ± 0.07 | 0.34 ± 0.09 |
| **Abstract 5** | imposing, impending, amazing, aspiring, invasive | 1.70 ± 0.24 | 0.05 ± 0.03 | 0.30 ± 0.05 |
| **Abstract 6** | coincidental, fundamental, temperamental, unsuccessful, accidentally | 1.79 ± 0.24 | 0.01 ± 0.04 | 0.32 ± 0.06 |
| **Abstract 7** | assume, astute, allure, acute, akin | 1.83 ± 0.09 | 0.04 ± 0.04 | 0.32 ± 0.09 |


## Acknowledgments
We would like to thank the authors of the following works:  
**Code References:**
- Choi, K., Pasad, A., Nakamura, T., Fukayama, S., Livescu, K., & Watanabe, S. (2024). *Self-Supervised Speech Representations are More Phonetic than Semantic*. arXiv preprint arXiv:2406.08619.
- Peng, P., & Harwath, D. (2022). *Fast-Slow Transformer for Visually Grounding Speech*. Proceedings of the 2022 International Conference on Acoustics, Speech and Signal Processing.

**Datasets and Embeddings:**
- Tucker, B. V., Brenner, D., Danielson, D. K., Kelley, M. C., Nenadić, F., & Sims, M. (2019). *The massive auditory lexical decision (MALD) database*. Behavior research methods, 51, 1187-1204.
- Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). *Librispeech: an ASR corpus based on public domain audio books*. 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5206-5210.
- Pennington, J., Socher, R., & Manning, C. D. (2014). *Glove: Global vectors for word representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532-1543.
- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). *Concreteness ratings for 40 thousand generally known English word lemmas*. Behavior research methods, 46, 904-911.
- Palaskar, S., Sanabria, R., & Metze, F. (2018). *End-to-end multimodal speech recognition*. 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5774-5778.
