# Databalancer

Databalancer is the python library using in machine learning applications to balance the imbalanced text classification datasets before the model training

<img src="https://raw.githubusercontent.com/pradeepdev-1995/databalancer/master/logo/logo.png" width="800" height="250">

## Features

* Databalancer is able to balance any imbalanced text classification datasets
* If the given dataset is imbalanced then while balancing no existing data will remove but new data will be generated and added to the dataset
* For a particular class the newly generated data will be the paraphrases of the existing data in that particular class
* By default these paraphrases are generated using the *ramsrigouthamg/t5_paraphraser* model (You can read more about the model from [Huggingface official documentation](https://huggingface.co/ramsrigouthamg/t5_paraphraser))
* Databalancer also provides another method called *classCountVisualization* to show the dataset class count distribution

## Installation

Install the `databalancer` package with `pip`

     pip install databalancer

## Compatibility

Databalancer is only compatable with python 3.6.9 or above.


## Quick Start
The library databalancer provides two different functionalities

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
balanceDataset("traindata.csv")
```

The above code will balance the dataset and store the balanced dataset(*'balanced_data.csv'*) in the local machine.

To show the balanced dataset class count distribution, run the code below

```python
from databalancer import classCountVisualization

classCountVisualization("balanced_data.csv")

```

![Balanced dataset pie plot](https://github.com/pradeepdev-1995/databalancer/blob/master/images/balancedDataset.png?raw=true "Balanced dataset pie plot")
