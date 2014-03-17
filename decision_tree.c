#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "decision_tree.h"

dt_node* dt_new_node();
void dt_free_node(dt_node *node);
int dt_split_on_node(dt_node *node, data_set *train_data, int depth, split_criterion criterion);
float dt_classify(decision_tree *dt, data_set *data, int row);
int count_nodes(dt_node *node);
float guess_node_class(decision_tree *dt, dt_node *node);

decision_tree* dt_new(unsigned int seed, split_criterion criterion) {
    if(seed > 0) {
        srand(seed);
    }
    else {
        srand(time(NULL));
    }

    decision_tree *dt = malloc(sizeof(decision_tree));
    dt->root = dt_new_node();
    dt->criterion = criterion;
    return dt;
}

void dt_free(decision_tree *dt) {
    dt_free_node(dt->root);
    free(dt);
}

int dt_node_count(decision_tree *dt) {
    return count_nodes(dt->root);
}

int dt_train(decision_tree *dt, data_set *train_data) {
    if(train_data->has_ydata == 0) {
        fprintf(stderr, "Data set must have y data!\n");
        return -1;
    }

    // this comes in handy occasionally
    dt->dataset = train_data;
    int count = dt_split_on_node(dt->root, train_data, 0, dt->criterion);
    printf("Decision tree has %d nodes\n", count);
    return 0;
}

float* dt_predict(decision_tree *dt, data_set *test_data) {
    float *preds = malloc(test_data->rowcount * sizeof(float));
    for(int i = 0; i < test_data->rowcount; i++) {
        float class = dt_classify(dt, test_data, i);
        preds[i] = class;
    }

    return preds;
}


// classify a single row in the data set
float dt_classify(decision_tree *dt, data_set *data, int row) {
    dt_node *node = dt->root;
    while(1) {
        if(node->is_leaf) {
            return node->prediction_value;
        }

        float value = data->x_data[row][node->split_col];
        if(value < node->split_value) {
            dt_node *tmpnode = node->left;
            if(tmpnode == NULL) {
                tmpnode = node->right;
                if(tmpnode == NULL) {
                    //fprintf(stderr, "Node is not a leaf, but has no children! Classifying as -1\n");
                    return 2;
                }
            }
            node = tmpnode;
        }
        else {
            dt_node *tmpnode = node->right;
            if(tmpnode == NULL) {
                tmpnode = node->left;
                if(tmpnode == NULL) {
                    //fprintf(stderr, "Node is not a leaf, but has no children! Classifying as -1\n");
                    return 2;
                }
            }
            node = tmpnode;
        }
    }
}

// compute the score for the validation data set
float dt_score(decision_tree *dt, data_set *validation_data) {
    if(!validation_data->has_ydata) {
        fprintf(stderr, "Scoring data must have y data!\n");
        return 0.0;
    }

    int total = validation_data->rowcount;
    int correct = 0;

    for(int i = 0; i < total; i++) {
        float class = dt_classify(dt, validation_data, i);
        float actual = validation_data->y_data[i];
        if(class == actual) {
            correct += 1;
        }
    }

    float ratio = ((float)correct) / total;
    return ratio;
}

dt_node* dt_new_node() {
    dt_node *node = malloc(sizeof(dt_node));
    node->is_leaf = 0;
    node->split_value = 0;
    node->prediction_value = 0;
    node->left = NULL;
    node->right = NULL;
    node->parent = NULL;
    return node;
}


void dt_free_node(dt_node *node) {
    if(node == NULL) {
        return;
    }
    dt_free_node(node->left);
    dt_free_node(node->right);
    free(node);
}

// pick the best column to split on, based on the information gain metric
int dt_pick_best_column(data_set *data, split_criterion criterion) {
    float *gains = malloc(data->colcount * sizeof(float));

    for(int col = 0; col < data->colcount; col++) {
        // divide up the data based on the mean of the chosen column
        float mean = ds_col_mean(data, col);

        data_set *lesser = ds_new(data->colcount, 1);
        data_set *greater = ds_new(data->colcount, 1);

        for(int row = 0; row < data->rowcount; row++) {
            if(data->x_data[row][col] < mean) {
                ds_add_item(lesser, data->x_data[row], data->y_data[row]);
            }
            else {
                ds_add_item(greater, data->x_data[row], data->y_data[row]);
            }
        }

        float main_splitscore;
        float lesser_splitscore;
        float greater_splitscore;

        if(criterion == CR_ENTROPY) {
            // entropy estimation for the whole data set and the two splits
            main_splitscore = ds_entropy(data);
            lesser_splitscore = ds_entropy(lesser);
            greater_splitscore = ds_entropy(greater);
        }
        else if(criterion == CR_GINI) {
            main_splitscore = ds_gini(data);
            lesser_splitscore = ds_gini(lesser);
            greater_splitscore = ds_gini(greater);
        }
        else {
            fprintf(stderr, "Unknown criterion %d!\n", criterion);
            return 0;
        }

        // ratios for split data sets
        float lesser_frac = ((float)lesser->rowcount) / data->rowcount;
        float greater_frac = ((float)greater->rowcount) / data->rowcount;

        // this is either information gain if the splitscore is entropy
        // or it is the total population diversity score if using gini
        float gain;
        if(criterion == CR_ENTROPY) {
            gain = main_splitscore - ((lesser_frac * lesser_splitscore) +
                    (greater_frac * greater_splitscore));
        }
        else if(criterion == CR_GINI) {
            gain = (lesser_frac * lesser_splitscore) +
                (greater_frac * greater_splitscore);
        }
        else {
            fprintf(stderr, "Unknown criterion %d!\n", criterion);
            return 0;
        }

        gains[col] = gain;
        ds_free(lesser);
        ds_free(greater);
    }

    // pick the best gain
    float best = gains[0];
    int bestcol = 0;
    for(int i = 0; i < data->colcount; i++) {
        if(gains[i] > best) {
            best = gains[i];
            bestcol = i;
        }
    }

    free(gains);
    return bestcol;
}


// returns 0 if all y values are the same
// 1 otherwise
int dt_should_split(data_set *data) {
    int last_seen = data->y_data[0];
    for(int i = 0; i < data->rowcount; i++) {
        if(data->y_data[i] != last_seen) {
            return 1;
        }
    }
    return 0;
}


int dt_split_on_node(dt_node *node, data_set *train_data, int depth, split_criterion criterion) {
    if(!dt_should_split(train_data)) {
        // all y values are the same, so make a leaf!
        node->is_leaf = 1;
        node->prediction_value = train_data->y_data[0];
        return 1;
    }
    else if(train_data->rowcount < 1) {
        // this is generally a bad place to be
        // should never happen
        fprintf(stderr, "No rows left in training set!\n");
        return 1;
    }

    // pick the best column based in info gain
    unsigned int col = dt_pick_best_column(train_data, criterion);

    // split on the mean of the column
    node->split_value = ds_col_mean(train_data, col);
    node->split_col = col;

    // make a new data set for all of the rows less than the mean
    data_set *lesser_data = ds_new(train_data->colcount, 1);

    // add all rows < mean
    for(int i = 0; i < train_data->rowcount; i++) {
        float val = train_data->x_data[i][col];
        if(val < node->split_value) {
            ds_add_item(lesser_data, train_data->x_data[i], train_data->y_data[i]);
        }
    }

    int c1 = 0;
    if(lesser_data->rowcount > 0) {
        // if we have data that was less than the mean (should always happen)
        // then recurse on that new data set
        dt_node *left_node = dt_new_node();
        left_node->is_lesser = 1;
        node->left = left_node;
        c1 = dt_split_on_node(left_node, lesser_data, depth+1, criterion);
    }
    else {
        node->left = NULL;
    }
    ds_free(lesser_data);

    // make a data set for values >= mean
    data_set *greater_data = ds_new(train_data->colcount, 1);

    for(int i = 0; i < train_data->rowcount; i++) {
        float val = train_data->x_data[i][col];
        if(val >= node->split_value) {
            ds_add_item(greater_data, train_data->x_data[i], train_data->y_data[i]);
        }
    }

    int c2 = 0;
    if(greater_data->rowcount > 0) {
        // recurse on the new data set
        dt_node *right_node = dt_new_node();
        node->right = right_node;
        right_node->is_lesser = 0;
        c2 = dt_split_on_node(right_node, greater_data, depth+1, criterion);
    }
    else {
        node->right = NULL;
    }
    ds_free(greater_data);

    // return a count of all of the decendent nodes for the current node
    return c1+c2;
}

// private function, returns a count of all children of the specified node plus
// the node itself (children + 1)
int count_nodes(dt_node *node) {
    if(node == NULL) {
        return 0;
    }
    return 1 + count_nodes(node->left) + count_nodes(node->right);
}

// this is a private function that recursively prunes nodes top-down
// and only accepts a pruning if it increases the prediction score of the
// validation data
// returns the number of nodes successfully pruned
int prune_node(decision_tree *dt, dt_node *node, data_set *validation_data) {
    // the score with both subtrees still attached
    float primary_score = dt_score(dt, validation_data);

    // save subtrees so that we can restore them if classification score
    // didn't improve
    dt_node *left = node->left;
    dt_node *right = node->right;
    int right_prune_count = 0;
    int left_prune_count = 0;

    if(left != NULL) {
        node->left = NULL;

        // score the decision tree with the missing subtree
        float left_prune_score = dt_score(dt, validation_data);
        if(left_prune_score >= primary_score) {
            // found a good prune!
            left_prune_count = count_nodes(left);
            float diff = left_prune_score - primary_score;
            if(diff > 0.0002 || left_prune_count > 10) {
                printf("Improved score by %.4f, dropped %d nodes\n",
                        diff, left_prune_count);
            }
            // throw away the subtree now that we don't need it
            dt_free_node(left);
        }
        else {
            // prune was no good, so restore the subtree and recurse
            node->left = left;
            left_prune_count = prune_node(dt, node->left, validation_data);
        }
    }

    if(right != NULL) {
        // basically the same as above, but for the right subtree
        node->right = NULL;

        float right_prune_score = dt_score(dt, validation_data);
        if(right_prune_score >= primary_score) {
            right_prune_count = count_nodes(right);
            float diff = right_prune_score - primary_score;
            if(diff > 0.0002 || right_prune_count > 10) {
                printf("Improved score by %.4f, dropped %d nodes\n",
                        diff, right_prune_count);
            }
            dt_free_node(right);
        }
        else {
            node->right = right;
            right_prune_count = prune_node(dt, node->right, validation_data);
        }
    }

    // need to see if we're a leaf now
    if(node->left == NULL && node->right == NULL) {
        node->prediction_value = guess_node_class(dt, node);
        node->is_leaf = 1;
    }

    return left_prune_count + right_prune_count;
}

// this is a public function for attempting to prune the decision tree and
// improve classification
int dt_prune(decision_tree *dt, data_set *validation_data) {
    return prune_node(dt, dt->root, validation_data);
}

// this is used by the pruning step. sometimes a node has both of its
// children pruned, so it needs to become a leaf node. this function
// returns the most common class of samples that end up at that leaf
float guess_node_class(decision_tree *dt, dt_node *leaf_node) {
    if(leaf_node->left != NULL || leaf_node->right != NULL) {
        fprintf(stderr, "Can't guess class of non-leaf node!\n");
        return 0;
    }

    if(leaf_node->is_leaf) {
        return leaf_node->prediction_value;
    }

    int found_root = 0;
    int classcount;
    float *classes = ds_classes(dt->dataset, &classcount);
    int *classcounts = malloc(classcount * sizeof(int));
    memset(classcounts, 0, classcount * sizeof(int));

    // to do this, we go from our node of choice, and try to see
    // if we can get to the root for each sample
    dt_node *curnode = leaf_node;
    for(int row = 0; row < dt->dataset->rowcount; row++) {
        while(curnode != NULL) {
            if(curnode == dt->root) {
                found_root = 1;
                // row was classified by our node! count its class
                for(int c = 0; c < classcount; c++) {
                    if(classes[c] == dt->dataset->y_data[row]) {
                        classcounts[c] += 1;
                        break;
                    }
                }
                break;
            }
            float split_val = curnode->split_value;
            unsigned int split_col = curnode->split_col;
            if(curnode->is_lesser) {
                if(dt->dataset->x_data[row][split_col] < split_val) {
                    curnode = curnode->parent;
                }
                else {
                    // the sample was not classified by our node of choice
                    // move on to the next sample
                    break;
                }
            }
            else {
                if(dt->dataset->x_data[row][split_col] >= split_val) {
                    curnode = curnode->parent;
                }
                else {
                    // the sample was not classified by our node of choice
                    // move on to the next sample
                    break;
                }
            }
        }
    }

    int best = classcounts[0];
    float bestclass = classes[0];
    for(int i = 0; i < classcount; i++) {
        if(classcounts[i] > best) {
            bestclass = classes[i];
            best = classcounts[i];
        }
    }

    if(!found_root) {
        fprintf(stderr, "Failed to find the root when guessing the class... (classcount was %d)\n", best);
    }

    free(classes);
    free(classcounts);
    return bestclass;
}


