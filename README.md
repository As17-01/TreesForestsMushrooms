# Trees, Forests, Mushrooms

Link to the competition: 

`https://www.kaggle.com/competitions/hse-ds-hw1-trees-forests-mushrooms/overview`

## Installation

Install the project and requirements with the following command:

`poetry install`

Enter the virtual environment using the following command:

`poetry shell`

## Steps to reproduce models:

### Download data

You can use `scripts/download/main.py` script to download the data from Kaggle. Or you can manually create `data` folder in the current directory and put `test.csv` and `train.csv` there. Also you should create `submissions` folder inside the `data` folder.

### Training

To train the submitted models you can write the following commands inside `scripts/train`:

* `python main.py data=decision_tree algorithm=decision_tree`. Time to execute: 5 seconds
* `python main.py data=random_forest algorithm=random_forest`. Time to execute: 4 minutes 49 seconds
* `python main.py data=ada_boost algorithm=ada_boost`. Time to execute: 3 minutes 38 seconds
