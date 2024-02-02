import pandas as pd
import math
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import copy
import argparse
start_time = time.time()

#make the node
class Node:
    def __init__(self, val, S0 = None, S1 = None, neg_list = None, pos_list = None):
        self.value = val
        self.left = S0
        self.right = S1
        self.neg_list = neg_list
        self.pos_list = pos_list
#grow tree
def grow_tree(S, E_parent, evflag):
    if (len(S["Y"].unique()) == 1) and (0 in S["Y"].unique()):
        return Node(0)
    elif (len(S["Y"].unique()) == 1) and (1 in S["Y"].unique()):
        return Node(1)
    # for the last feature remaining just split on it and give its leaf nodes 0 or 1 value
    elif len(S.columns) <= 2:
        temp_zero = S["Y"][S.iloc[:, 0] == 0]
        zero_ones = temp_zero.sum()
        zero_zeros = len(temp_zero) - zero_ones
        neg_list = [zero_zeros, zero_ones]

        temp_one = S["Y"][S.iloc[:, 0] == 1]
        one_ones = temp_one.sum()
        one_zeros = len(temp_one) - one_ones
        pos_list = [one_zeros, one_ones]
        # if equal no. of 1 and 0 exist in the target column then the value will be 0
        if zero_ones > zero_zeros:
            left = 1
        else:
            left = 0
        if one_ones > one_zeros:
            right = 1
        else:
            right = 0
        return Node(S.columns[0], Node(left), Node(right),neg_list, pos_list )
    else:
        best_attribute, E0, E1, neg_list, pos_list = info_gain(S, E_parent, evflag)
        S0 = S[S[best_attribute] == 0]
        S1 = S[S[best_attribute] == 1]
        if len(S0.columns) > 2:
            S0 = S0.drop([best_attribute], axis=1)
        if len(S1.columns) > 2:
            S1 = S1.drop([best_attribute], axis =1)

        return Node(best_attribute, grow_tree(S0, E0, evflag), grow_tree(S1, E1, evflag), neg_list, pos_list)
#calculate entropy
def entropy(no_neg, no_pos):
    total = no_pos + no_neg
    if no_pos == 0 or no_neg == 0:
        E = 0
    else:
        E = - (float(float(no_pos)/float(total))) * math.log(float((float(no_pos)/float(total))),2) - (float(float(no_neg)/float(total))) * math.log(float((float(no_neg)/float(total))),2)
    return E
#calculate variance
def variance(no_neg, no_pos):
    total = no_pos + no_neg
    if no_pos == 0 or no_neg == 0:
        V = 0
    else:
        V = float(float((no_neg*no_pos))/float((total*total)))
    return V
#calculating info gain
def info_gain(S, E, evflag):
    best_attribute = None
    max_gain = 0
    zero_list = [0, 0]
    one_list = [0, 0]
    E_pos = 0
    E_neg = 0
    S_arr_t = S.values.T
    for col in S_arr_t[:-1]:
        neg_list = [0, 0]
        pos_list = [0, 0]
        for i in range(len(col)):
            if col[i] == 0 and S_arr_t[-1][i] == 0:
                neg_list[0] = neg_list[0] + 1
            if col[i] == 0 and S_arr_t[-1][i] == 1:
                neg_list[1] = neg_list[1] + 1
            if col[i] == 1 and S_arr_t[-1][i] == 0:
                pos_list[0] = pos_list[0] + 1
            if col[i] == 1 and S_arr_t[-1][i] == 1:
                pos_list[1] = pos_list[1] + 1

        if evflag == 0:
            E0 = entropy(neg_list[0], neg_list[1])
            E1 = entropy(pos_list[0], pos_list[1])
        else:
            E0 = variance(neg_list[0], neg_list[1])
            E1 = variance(pos_list[0], pos_list[1])

        total = neg_list[0] + neg_list[1] + pos_list[0] + pos_list[1]
        G = E - ((pos_list[0] + pos_list[1]) / total) * E1 - ((neg_list[0] + neg_list[1]) / total) * E0

        if max_gain < G:
            max_gain = G
            best_attribute = S.columns[S_arr_t.tolist().index(col.tolist())]
            E_neg = E0
            E_pos = E1
            zero_list = neg_list
            one_list = pos_list

    return (best_attribute, E_neg, E_pos, zero_list, one_list)
#traverse the tree that has grown
def traverse_tree(d_tree, row):
    if d_tree.value == 0 :
        return 0
    elif d_tree.value == 1:
        return 1
    else:
        index = d_tree.value.replace("X", "")
        val = row[int(index)]
        if val == 0:
           return traverse_tree(d_tree.left, row)
        else:
            return traverse_tree(d_tree.right, row)
#test the tree on the test class
def label_maker(d_tree, t_set):
    label_list = []
    np_mat = t_set.iloc[:, 0:-1].values
    for row in np_mat.tolist():
        label_list.append(traverse_tree(d_tree, row))
    label_ser = pd.Series({"Pred_Y" : label_list})
    correct = (t_set.Y == label_ser.Pred_Y).sum()

    acu = correct / len(label_ser.Pred_Y)
    return acu, label_ser

def max_depth(node):
    if node is None:
        return 0
    else:
        l_depth = max_depth(node.left)
        r_depth = max_depth(node.right)

        if l_depth > r_depth:
            return l_depth + 1
        else:
            return r_depth + 1
level_nodes = []
def bfs(d_tree):
    level_list = []
    d = max_depth(d_tree)
    for i in range(1, d + 1):
        add_level(d_tree, i)
        temp_l = copy.copy(level_nodes)
        level_list.append(temp_l)
        level_nodes.clear()
    return level_list

def add_level(d_tree, level):
    if d_tree is None:
        return
    if level == 1:
        level_nodes.append(d_tree)
    elif level > 1:
        add_level(d_tree.left, level - 1)
        add_level(d_tree.right, level - 1)
#pruning algorithm
def re_prune(level_list, v_set, d_tree, accuracy):
    prev_accuracy = accuracy
    for level in level_list[:-1]:
        for node in level:
            if node.value != 0 and node.value != 1:
                temp_value = node.value
                temp_left = node.left
                temp_right = node.right
                zeros = node.neg_list[0] + node.pos_list[0]
                ones = node.neg_list[1] + node.pos_list[1]
                if ones > zeros:
                    node.value = 1
                else:
                    node.value = 0
                node.left = None
                node.right = None
                accu, lbl_set = label_maker(d_tree, v_set)
                if accu < prev_accuracy:
                    node.value = temp_value
                    node.left = temp_left
                    node.right = temp_right
                else:
                    prev_accuracy = accu
    return prev_accuracy

#depth based pruning algorithm
def db_prune(l_list, s_d_list, v_set, d_tree, accuracy):
    acc_list = []
    for d in s_d_list:
        store_list = []
        count = 0
        if d < len(l_list):
            for n in l_list[d]:
                if (n.value != 0) and (n.value != 1):
                    a = copy.deepcopy(n)
                    idx = l_list[d].index(n)
                    store_list.append((a, idx))
                    if (n.neg_list[1] + n.pos_list[1]) > (n.neg_list[0] + n.pos_list[0]):
                        n.value = 1
                    else:
                        n.value = 0
                    n.left = None
                    n.right = None
            acu, lbl_set = label_maker(d_tree, v_set)
            acc_list.append(acu)
            print("max depth of tree = {}".format(len(l_list) - 1))
            print("accuracy on validation set = {}, depth = {}".format(acu, d))
            print("percentage change in accuracy w.r.t naive = {}".format(acu - accuracy))



            for a, i in store_list:
                l_list[d][i].value = a.value
                l_list[d][i].left = a.left
                l_list[d][i].right = a.right

    max_d = s_d_list[acc_list.index(max(acc_list))]
    for n in l_list[max_d]:
        if (n.value != 0) and (n.value != 1):
            if (n.neg_list[1] + n.pos_list[1]) > (n.neg_list[0] + n.pos_list[0]):
                n.value = 1
            else:
                n.value = 0
            n.left = None
            n.right = None
    return max_d, max(acc_list)


def main():

    parser = argparse.ArgumentParser(description = "Runs the specified algorithm on the file path mentioned")
    parser.add_argument('-algorithm_number', '--algo_no', type=int)
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-valid_data', '--valid_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    algo_no = arg.algo_no
    train_name = arg.train_set
    valid_name = arg.valid_set
    test_name = arg.test_set

    train_set = pd.read_csv(train_name, header = None)
    valid_set = pd.read_csv(valid_name, header = None)
    test_set = pd.read_csv(test_name, header= None)

    for el in [train_set,valid_set, test_set]:
        ncols = len(el.columns)
        nrows = len(el.index)
        el.columns = ["X{}".format(i) for i in range(ncols)]
        l = list(el.columns)
        l[-1] = "Y"
        el.columns = l

    Y = train_set.iloc[:, len(train_set.columns) - 1]
    no_pos_Y = int(Y.sum())
    no_neg_Y = int(len(train_set.index) - no_pos_Y)
    entropy_Y = entropy(no_pos_Y, no_neg_Y)
    variance_Y = variance(no_pos_Y, no_neg_Y)

    if algo_no == 1:
        print("Running naive decision tree with entropy heuristic...")
        evflag = 0
        print("Growing tree...")
        d_tree = grow_tree(train_set, entropy_Y, evflag)
        print("Tree grown.")
        print("Finding accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The accuracy on test_set = {}".format(accuracy))
    elif algo_no == 2:
        print("Running naive decision tree with variance heuristic...")
        evflag = 1
        print("Growing tree...")
        d_tree = grow_tree(train_set, variance_Y, evflag)
        print("Tree grown.")
        print("Finding accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The accuracy on test_set = {}".format(accuracy))
    elif algo_no == 3:
        print("Running decision tree with entropy heuristic and reduced error pruning...")
        evflag = 0
        print("Growing tree...")
        d_tree = grow_tree(train_set, entropy_Y, evflag)
        print("Tree grown.")
        print("Finding naive accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The naive accuracy on test_set = {}".format(accuracy))
        level_list = bfs(d_tree)
        level_list.reverse()
        print("Pruning the tree...")
        re_accuracy = re_prune(level_list, valid_set, d_tree, accuracy)
        post_pruning_accuracy, labels = label_maker(d_tree, test_set)
        print("Accuracy on valid_set after reduced error pruning = {}".format(re_accuracy))
        print("change in accuracy w.r.t Naive = {}".format(re_accuracy - accuracy))
        print("post RE pruning accuracy on test_set = {}".format(post_pruning_accuracy))
        print("change in post pruning accuracy w.r.t Naive = {}".format(post_pruning_accuracy - accuracy))
    elif algo_no == 4:
        print("Running decision tree with variance heuristic and reduced error pruning...")
        evflag = 1
        print("Growing tree...")
        d_tree = grow_tree(train_set, variance_Y, evflag)
        print("Tree grown.")
        print("Finding naive accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The naive accuracy on test_set = {}".format(accuracy))
        level_list = bfs(d_tree)
        level_list.reverse()
        print("Pruning the tree...")
        re_accuracy = re_prune(level_list, valid_set, d_tree, accuracy)
        post_pruning_accuracy, labels = label_maker(d_tree, test_set)
        print("Accuracy on valid_set after reduced error pruning = {}".format(re_accuracy))
        print("change in accuracy w.r.t Naive = {}".format(re_accuracy - accuracy))
        print("post RE pruning accuracy on test_set = {}".format(post_pruning_accuracy))
        print("change in post pruning accuracy w.r.t Naive = {}".format(post_pruning_accuracy - accuracy))
    elif algo_no == 5:
        print("Running decision tree with entropy heuristic and depth based pruning...")
        evflag = 0
        print("Growing tree...")
        d_tree = grow_tree(train_set, entropy_Y, evflag)
        print("Tree grown.")
        print("Finding naive accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The naive accuracy on test_set = {}".format(accuracy))
        stop_depths = [5, 10, 15, 20, 50, 100]
        level_list = bfs(d_tree)
        d_max, db_accuracy = db_prune(level_list, stop_depths, valid_set, d_tree, accuracy)
        post_dbpruning_accuracy, label_db = label_maker(d_tree, test_set)
        print("Max depth = {}, accuracy = {}".format(d_max, db_accuracy))
        print("post pruning accuracy on test set: {} ".format(post_dbpruning_accuracy))
        print("percent change in accuracy w.r.t Naive = {}".format(post_dbpruning_accuracy - accuracy))
    elif algo_no == 6:
        print("Running decision tree with variance heuristic and depth based pruning...")
        evflag = 1
        print("Growing tree...")
        d_tree = grow_tree(train_set, variance_Y, evflag)
        print("Tree grown.")
        print("Finding naive accuracy on the test_set...")
        accuracy, labels = label_maker(d_tree, test_set)
        print("The naive accuracy on test_set = {}".format(accuracy))
        stop_depths = [5, 10, 15, 20, 50, 100]
        level_list = bfs(d_tree)
        d_max, db_accuracy = db_prune(level_list, stop_depths, valid_set, d_tree, accuracy)
        post_dbpruning_accuracy, label_db = label_maker(d_tree, test_set)
        print("Max depth = {}, accuracy = {}".format(d_max, db_accuracy))
        print("post pruning accuracy on test set: {} ".format(post_dbpruning_accuracy))
        print("percent change in accuracy w.r.t Naive = {}".format(post_dbpruning_accuracy - accuracy))

    elif algo_no == 7:
        rfc = RandomForestClassifier(n_estimators= 100, random_state= 1)
        rfc.fit(train_set.iloc[:, 0:-1].values,train_set["Y"].values)
        # predictions
        rfc_predict = rfc.predict(test_set.iloc[:, 0:-1].values)
        print("RF Accuracy:",metrics.accuracy_score(test_set["Y"].values, rfc_predict))

    print("Time taken = {}".format(time.time() - start_time))

main()