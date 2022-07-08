# Databalancer

Databalancer is the python library using in machine learning applications to balance the imbalanced text classification datasets before the model training.

<img src="https://raw.githubusercontent.com/pradeepdev-1995/databalancer/master/logo/logo.png" width="800" height="250">

## Features

* Databalancer is able to balance any imbalanced text classification datasets
* If the given dataset is imbalanced then while balancing no existing data is removed, but new data will be generated and added to the dataset
* For a particular class the newly generated data will be the paraphrases of the existing data in that particular class
* By default, these paraphrases are generated using the *ramsrigouthamg/t5_paraphraser* model (You can read more about the model from [Huggingface official documentation](https://huggingface.co/ramsrigouthamg/t5_paraphraser))
* The current version can generate the sentence paraphrases using multiple methods such as T5 models, NLPAUG and Textattack
* The user can select the balance method by passing the `balance_method` parameter while calling the `balanceDataset` method such as
  * `balance_method=1` for `ramsrigouthamg/t5_paraphraser` T5 model based balancing (Default) ( For more info check [t5_paraphraser](https://huggingface.co/ramsrigouthamg/t5_paraphraser))
  * `balance_method=2` for `ramsrigouthamg/t5-large-paraphraser-diverse-high-quality` T5 model based balancing (For more info check [t5-large-paraphraser-diverse-high-quality](https://huggingface.co/ramsrigouthamg/t5-large-paraphraser-diverse-high-quality))
  * `balance_method=3` for `nlpaug` based balancing (For more info check [nlpaug](https://pypi.org/project/nlpaug/))
  * `balance_method=4` for `textattack` based balancing (For more info check [textattack](https://pypi.org/project/textattack/))
* The `model` argument in the `balanceDataset` method is only applicable when `balance_method` is set as `3`, through which user can pass the transformer model name from [Huggingface](https://huggingface.co/models) to generate paraphrases using NLPAUG .
* If the user enable `quantize=True` in `balanceDataset` then the T5 models(`balance_method==1` and `balance_method=2`) will go through the quantization process using [fastT5](https://pypi.org/project/fastt5/) before inference, so that the model inference time will be reduced.
* By default `quantize` parameter is set as `False` because quantization requires more RAM and more CPU Processing power
* Databalancer also provides another method called *classCountVisualization* to show the dataset class count distribution

## Installation

Install the `databalancer` package with `pip`

     pip install databalancer

## Compatibility

Databalancer is only compatable with python 3.6.9 or above.


## Quick Start
The library databalancer provides two different functionalities.

1 - classCountVisualization

2 - balanceDataset

### classCountVisualization

```python
#Import the classCountVisualization from the 'databalancer' module
from databalancer import classCountVisualization
    
#Pass the required datasetname(here traindata.csv) to the function
classCountVisualization("traindata.csv")

```

### Output

![Imbalanced dataset pie plot](https://raw.githubusercontent.com/pradeepdev-1995/databalancer/master/images/imbalancedDatset.png?raw=true "Imbalanced dataset pie plot")

### balanceDataset
```python
#Import the balanceDataset from the 'databalancer' module
from databalancer import balanceDataset

#Pass the dataset name which is to be balanced(here traindata.csv) to the balanceDataset function
balanceDataset("traindata.csv",balance_method=1)
```

The above code will balance the dataset and store the balanced dataset(*'balanced_data.csv'*) in the local machine.

### balanceDataset with model quantization 

```python
#Import the balanceDataset from the 'databalancer' module
from databalancer import balanceDataset

#Pass the dataset name which is to be balanced(here traindata.csv) to the balanceDataset function with balance_method=2 and enable quantization 
balanceDataset("traindata.csv",balance_method=2,quantize=True)
```

The above code will balance the dataset using balance_method=2 with quantization and store the balanced dataset(*'balanced_data.csv'*) in the local machine.

To show the balanced dataset class count distribution, run the code below.

```python
from databalancer import classCountVisualization

classCountVisualization("balanced_data.csv")

```

![Balanced dataset pie plot](https://github.com/pradeepdev-1995/databalancer/blob/master/images/balancedDataset.png?raw=true "Balanced dataset pie plot")
