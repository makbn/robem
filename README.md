# ROBEM

The paradigm of fine-tuning Pre-trained Language Models (PLMs) has been successful in Entity Matching (EM). Despite their remarkable performance, PLMs exhibit tendency to learn spurious correlations from training data. In this work, we aim at investigating whether PLM-based entity matching models can be trusted in real-world applications where data distribution is different from that of training. To this end, we design an evaluation benchmark to assess the robustness of EM models to facilitate their deployment in the real-world settings.
Our assessments reveal that data imbalance in the training data is a key problem for robustness. We also find that data augmentation alone is not sufficient to make a model robust. As a remedy, 
we prescribe simple modifications that can improve the robustness of PLM-based EM models. Our experiments show that while yielding superior results for in-domain generalization, our proposed model significantly improves the model robustness, compared to state-of-the-art EM models.


For more technical details, see the Probing the Robustness of Pre-trained Language Models for
Entity Matching paper.

# Requirements

Check out the [requirement file](https://github.com/makbn/robem/blob/master/requirements.txt) to see the libraries and versions. To install all requirements:

```bash
pip install -r requirements.txt
```

Check the [Configuration](https://github.com/makbn/robem/blob/master/em/config.py) to see all available flags and hyperparameters.



* Dataset: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
* Ditto (ditto_light) package: https://github.com/megagonlabs/ditto/tree/master/ditto_light

* Arguments


| **Arg**        | **Description**                                                           | **Default**    | **Values**                                                                                                                |
|----------------|---------------------------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------|
| --dataset_name | select dataset                                                            | itunes-amazon  | 'beer-rates', 'itunes-amazon', 'amazon-google', 'abt-buy', 'fodors-zagats' , 'dblp-acm', 'dblp-scholar', 'walmart-amazon' |
| --lm           | base language model                                                       | roberta-base   | 'bert', 'bert-large', 'roberta-base'                                                                                      |
| --lr           | learning rate                                                             | 3e-5           | any valid number                                                                                                          |
| --da           | enable simple data augmentation, without this flag, simple da is disabled | False          | True/False                                                                                                                |
| --ditto_aug    | enable ditto data augmentation                                            | all            | 'del', 'drop_col', 'append_col', 'drop_token', 'drop_len',             'drop_sym', 'drop_same', 'swap', 'ins', 'all'      |
| --deep         | enable deep classifier                                                    | True           | True/False                                                                                                                |
| --addsep       | add attribute separator as special token for to tokenizer and model(LM)   | False          | True/False                                                                                                                |
| --wd           | weight decay                                                              | 0              | any valid number                                                                                                          |
| --save_dir     | save directory for model checkpoint                                       | ../checkpoint/ | any valid directory                                                                                                       |
| --neg_weight   | wce & asl loss weight for non-match samples                               | 0.20           | a valid number between 0.00 and 1.00                                                                                      |
| --pos_weight   | wce & asl loss weight for positive samples                                | 0.80           | a valid number between 0.00 and 1.00                                                                                      |

Check `em/config.py` for the complete list.

# Citation

Please cite our related paper if you used this code as a part of your research.

```bibtex
@inproceedings{robem_22,
  author = {Akbarian Rastaghi, Mehdi and Kamalloo, Ehsan and Rafiei, Davood},
  title = {Probing the Robustness of Pre-Trained Language Models for Entity Matching},
  year = {2022},
  isbn = {9781450392365},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3511808.3557673},
  doi = {10.1145/3511808.3557673},
  abstract = {The paradigm of fine-tuning Pre-trained Language Models (PLMs) has been successful in Entity Matching (EM). Despite their remarkable performance, PLMs exhibit tendency to learn spurious correlations from training data. In this work, we aim at investigating whether PLM-based entity matching models can be trusted in real-world applications where data distribution is different from that of training. To this end, we design an evaluation benchmark to assess the robustness of EM models to facilitate their deployment in the real-world settings. Our assessments reveal that data imbalance in the training data is a key problem for robustness. We also find that data augmentation alone is not sufficient to make a model robust. As a remedy, we prescribe simple modifications that can improve the robustness of PLM-based EM models. Our experiments show that while yielding superior results for in-domain generalization, our proposed model significantly improves the model robustness, compared to state-of-the-art EM models.},
  booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  pages = {3786–3790},
  numpages = {5},
  keywords = {named entity disambiguation, entity matching, entity linking},
  location = {Atlanta, GA, USA},
  series = {CIKM '22}
}
```

# Authors

* [Mehdi Akbarian Rastaghi](https://www.linkedin.com/in/mehdiakbarian/)
* [Ehsan Kamalloo](https://webdocs.cs.ualberta.ca/~kamalloo/)
* [Davood Rafiei](https://webdocs.cs.ualberta.ca/~drafiei/)
