# A Deep Dive into Decision Trees and Random Forests

## A breakdown of the project including objectives and learning outcomes available on my [website](https://petrichor1998.github.io)

The zip file contains the code for the problem that has been stated in the pdf labeled "hw1"

code submitted by: PARTH PADALKAR. UTD ID 2021473758

language : Python2.7
library requirements:
Pandas
numpy
scikit-learn

------------------------------------------------------------------------------------------------
The program runs the algorithms one after the other

#commandline arguments:

-train_data : used to input the training data path
-test_data : used to input the testing data path
-valid_data : used to input the validataion data path
-alorithm_number : used to input the algorithm that you wish to run on the data

1: naive decision tree with entropy heuristic
2: naive decision tree with variance heuristic
3: decision tree with entropy heuristic and reduced error pruning
4: decision tree with variance heuristic and reduced error pruning
5: decision tree with entropy heuristic and depth based pruning
6: decision tree with variance heuristic and depth based pruning
7: Random forset

---------------------------------------------------------------------------------------------------
example command 

python d_tree.py -algorithm_number 2 -train_data "all_data/train_c300_d100.csv" -valid_data "all_data/valid_c300_d100.csv" -test_data "all_data/test_c300_d100.csv"
