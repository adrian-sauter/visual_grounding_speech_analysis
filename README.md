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
