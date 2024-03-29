

Spoken Language Understanding
------------------------------------
* Audio 입력을 인식하여 Intent를 예측하는 End2End Spoken Langauge Understanding
------------------------------------

Table of Contents
------------------------------------

<!--ts-->
   * [Installation](#installation)
   * [Dataset](#dataset)
   * [Model](#model)
   * [Reference](#reference)
<!--te-->

------------------------------------

Installation
------------------------------------

* **Python** >= 3.6
* **PyTorch** version == 1.4
* ...
* For pre-training models, you'll also need high-end GPU(s).

```bash=
git clone https://github.com/rainmaker712/spoken_langauge_understanding
pip install -r requirements.txt
```

------------------------------------

Dataset
------------------------------------
[Fluent Command Speech Dataset, need auth](https://groups.google.com/a/fluent.ai/forum/#!forum/fluent-speech-commands)

------------------------------------

Model
------------------------------------

[Back to Top](#table-of-contents)

------------------------------------

How to train
------------------------------------
1. Download Fluent Speech Dataset
2. Run slu_train.py

```Python=
# example
python slu_train.py --data-path fluent_speech_commands_dataset
```

[Back to Top](#table-of-contents)

------------------------------------
Reference
------------------------------------
* [Pytorch](https://github.com/pytorch/pytorch)
* [Transformers](https://github.com/huggingface/transformers)
* [espnet](https://github.com/espnet/espnet)

[Back to Top](#table-of-contents)