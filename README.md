# DziriBERT

<img src="https://github.com/alger-ia/dziribert/blob/main/dziribert_drawing.png" alt="dziribert drawing" width="17%" height="17%" align="right"/>

DziriBERT is the first Transformer-based Language Model that has been pre-trained specifically for the Algerian Dialect. It handles Algerian text contents written using both Arabic and Latin characters. It sets new state of the art results on Algerian text classification datasets (see below).

The model is publicly available at: https://huggingface.co/alger-ia/dziribert.

For more information, please visit our paper: https://arxiv.org/pdf/2109.12346.pdf

**\* New: \*** A fine-tuned version for sentiment classification available [here](https://huggingface.co/alger-ia/dziribert_sentiment).

## Evaluation

Results on the [Twifil dataset](https://aclanthology.org/2020.lrec-1.151/):

<center>
  
|            Model                         | Sentiment acc. | Emotion acc. |
| ---------------------------------------- |:--------------:|:------------:|
| bert-base-multilingual-cased             |      74.2 %    |    62.0 %    |
| aubmindlab/bert-base-arabert             |      73.9 %    |    64.6 %    |
| CAMeL-Lab/bert-base-arabic-camelbert-mix |      77.7 %    |    69.1 %    |
| qarib/bert-base-qarib                    |      78.8 %    |    68.9 %    |
| UBC-NLP/MARBERT                          |      80.5 %    |    70.2 %    |
| alger-ia/dziribert                       |      80.5 %    |    70.4 %    |

</center>


Results on the [Narabizi dataset](https://aclanthology.org/2021.findings-acl.324.pdf):

<center>
  
|            Model                         | Sentiment acc. |  Topic acc.  |
| ---------------------------------------- |:--------------:|:------------:|
| bert-base-multilingual-cased             |      52.6 %    |    49.3 %    |
| aubmindlab/bert-base-arabert             |      49.1 %    |    42.8 %    |
| CAMeL-Lab/bert-base-arabic-camelbert-mix |      49.4 %    |    47.0 %    |
| qarib/bert-base-qarib                    |      55.0 %    |    45.7 %    |
| UBC-NLP/MARBERT                          |      58.0 %    |    49.0 %    |
| alger-ia/dziribert                       |      63.5 %    |    62.8 %    |

</center>

In order to reproduce these results, please install the following requirements:  

```bash
pip install -r requirements.txt
```

Then, run the following evaluation script:

```bash
python3 evaluate_model.py
```

These results have been obtained on a Tesla K80 GPU.

## Pretrained DziriBERT

DziriBERT has been uploaded to the HuggingFace hub in order to facilitate its use: https://huggingface.co/alger-ia/dziribert.

It can be easily downloaded and loaded using the [transformers library](https://github.com/huggingface/transformers):

```python
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("alger-ia/dziribert")
model = BertForMaskedLM.from_pretrained("alger-ia/dziribert")

```

## Limitations

The pre-training data used in this project comes from social media (Twitter). Therefore, the Masked Language Modeling objective may predict offensive words in some situations. Modeling this kind of words may be either an advantage (e.g. when training a hate speech model) or a disadvantage (e.g. when generating answers that are directly sent to the end user). Depending on your downstream task, you may need to filter out such words especially when returning automatically generated text to the end user. 

## How to cite

```bibtex
@article{dziribert,
  title={DziriBERT: a Pre-trained Language Model for the Algerian Dialect},
  author={Abdaoui, Amine and Berrimi, Mohamed and Oussalah, Mourad and Moussaoui, Abdelouahab},
  journal={arXiv preprint arXiv:2109.12346},
  year={2021}
}
```

## Contact 

Please contact amine.abdaoui.nlp@gmail.com for any question, feedback or request.
