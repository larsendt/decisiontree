#pragma once

#include "data_set.h"

typedef enum split_criterion {
    CR_GINI,
    CR_ENTROPY
} split_criterion;

typedef struct dt_node {
    float split_value;
    unsigned int split_col;
    int is_leaf;
    int is_lesser;
    float prediction_value;
    struct dt_node *left;
    struct dt_node *right;
    struct dt_node *parent;
} dt_node;

typedef struct decision_tree {
    dt_node *root;
    data_set *dataset;
    split_criterion criterion;
} decision_tree;

// create a new decision tree
// if seed is zero, the current time will be used instead
// criterion should be one of CR_GINI or CR_ENTROPY
// criterion determines what split metric should be used in training
// on average, CR_GINI (population diversity) seems to perform better
decision_tree* dt_new(unsigned int seed, split_criterion criterion);

// free all memory associated with the decision tree
// dt should not be used after calling this function
void dt_free(decision_tree *dt);

// return the number of nodes in the tree
int dt_node_count(decision_tree *dt);

// train the decision tree on the given data set
// train_data REQUIRES Y data.
int dt_train(decision_tree *dt, data_set *train_data);

// return an array of predicted classes for test_data
// the length of the array is equal to the number of rows in test_data
// the array should be freed after use (bad C style, I know)
float* dt_predict(decision_tree *dt, data_set *test_data);

// return a scoring value based on how accurate the predictions
// for validation_data were. validation_data REQUIRES Y data.
// the return value is 1.0 for perfect prediction, and 0.0 if none of the
// samples were predicted correctly
float dt_score(decision_tree *dt, data_set *validation_data);

// attempt to prune the decision tree to improve classification accuracy on the
// validation data. this function is not automatically called, and may run for
// a long time
// returns the number of nodes pruned
int dt_prune(decision_tree *dt, data_set *validation_data);



