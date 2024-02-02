code submitted by: PARTH PADALKAR. UTD ID 2021473758

language : Python3.7
library requirements:
Pandas: version 0.25.1
numpy: version 1.16.4
scikit-learn: version 0.21.2

------------------------------------------------------------------------------------------------
The program takes the algorithm number and the address of the data files on your system.
The algorithm reference number is as follows:

1 - naive decision tree with entropy heuristic
2 - naive decision tree with variance heuristic
3 - decision tree with entropy heuristic and reduced error pruning
4 - decision tree with variance heuristic and reduced error pruning
5 - decision tree with entropy heuristic and depth based pruning
6 - decision tree with variance heuristic and depth based pruning
7 - Random forest

The path of the file should be given as a string. For example:
-to run test_c300_d100.csv, give "all_data/test_c300_d100.csv" as parameter

#commandline arguments:

-algorithm_number : used to specify the number of the algorithm to be implemented (refer above list)
-train_data : used to input the training file address as a string
-valid_data : used to input the validation file address as a string
-test_data : used to input the testing file address as a string

---------------------------------------------------------------------------------------------------
example command to run "naive decision tree with variance heuristic" algorithm on the "test_c300_d100.csv", "train_c300_d100.csv", "valid_c300_d100.csv" files:

python d_tree.py algorithm_number 2 -train_data "all_data/train_c300_d100.csv" -valid_data "all_data/valid_c300_d100.csv" -test_data "all_data/test_c300_d100.csv"



