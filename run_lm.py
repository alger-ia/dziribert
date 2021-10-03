from transformers import pipeline

examples = [" أنا من الجزائر من ولاية [MASK] ",
            "rabi [MASK] khouya sami",
            " ربي [MASK] خويا لعزيز"]


fill_mask = pipeline(
    "fill-mask",
    model="alger-ia/dziribert",
    tokenizer="alger-ia/dziribert"
)

for example in examples:
    print("--------------------------------------")
    print(example)
    print("--------------------------------------")
    for pred in fill_mask(example):
        print(pred['sequence'], pred['score'], pred['token_str'])
