# DziriBERT

<img src="https://github.com/alger-ia/dziribert/blob/main/dziribert_drawing.png" alt="dziribert drawing" width="15%" height="15%" align="right"/>

DziriBERT is the first Transformer-based Language Model that has been pre-trained specifically for the Algerian Dialect. It handles Algerian text contents written using both Arabic and Latin characters. It sets new state of the art results on Algerian text classification datasets, even if it has been pre-trained on much less data (~1 million tweets).

The model is publicly available at: https://huggingface.co/alger-ia/dziribert.

## Evaluation

The [Twifil dataset](https://aclanthology.org/2020.lrec-1.151/) was used to compare DziriBERT with current multilingual, standard Arabic and dialectal Arabic models:

<center>
  
|            Model                         | Sentiment acc. | Emotion acc. |
| ---------------------------------------- |:--------------:|:------------:|
| bert-base-multilingual-cased             |      73.6 %    |    59.4 %    |
| aubmindlab/bert-base-arabert             |      72.1 %    |    61.2 %    |
| CAMeL-Lab/bert-base-arabic-camelbert-mix |      77.1 %    |    65.7 %    |
| qarib/bert-base-qarib                    |      77.7 %    |    67.6 %    |
| UBC-NLP/MARBERT                          |      80.1 %    |    68.4 %    |
| alger-ia/dziribert                       |      80.3 %    |    69.3 %    |

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

## How to cite

```bibtex
@article{dziribert,
  title={DziriBERT: a Pre-trained Language Model for the Algerian Dialect},
  author={Abdaoui, Amine and Berrimi, Mohamed and Oussalah, Mourad and Moussaoui, Abdelouahab},
  journal={arXiv preprint arXiv:XXX.XXXXX},
  year={2021}
}
```

## Contact 

Please contact amine.abdaoui@huawei.com for any question, feedback or request.
