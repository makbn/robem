# ROBEM

The paradigm of fine-tuning Pre-trained Language Models (PLMs) has been successful in Entity Matching (EM). Despite their remarkable performance, PLMs exhibit tendency to learn spurious correlations from training data. In this work, we aim at investigating whether PLM-based entity matching models can be trusted in real-world applications where data distribution is different from that of training. To this end, we design an evaluation benchmark to assess the robustness of EM models to facilitate their deployment in the real-world settings.
Our assessments reveal that data imbalance in the training data is a key problem for robustness. We also find that data augmentation alone is not sufficient to make a model robust. As a remedy, 
we prescribe simple modifications that can improve the robustness of PLM-based EM models. Our experiments show that while yielding superior results for in-domain generalization, our proposed model significantly improves the model robustness, compared to state-of-the-art EM models.


For more technical details, see the Probing the Robustness of Pre-trained Language Models for
Entity Matching paper.


* Dataset: https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md
* Ditto (ditto_light) package: https://github.com/megagonlabs/ditto/tree/master/ditto_light

# Authors

* [Mehdi Akbarian Rastaghi](https://www.linkedin.com/in/mehdiakbarian/)
* [Ehsan Kamalloo](https://webdocs.cs.ualberta.ca/~kamalloo/)
* [Davood Rafiei](https://webdocs.cs.ualberta.ca/~drafiei/)