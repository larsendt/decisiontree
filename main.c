#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "csv.h"
#include "data_set.h"
#include "decision_tree.h"

int main(int argc, char *argv[]) {
    printf("Decision tree!\n");

    if(argc != 7) {
        fprintf(stderr, "Usage: %s [entropy|gini] [prune|noprune] <train csv> <valiate csv> <test csv> <prediction file>\n", argv[0]);
        return 1;
    }

    char *split_metric = argv[1];
    split_criterion criterion;
    char *prune_str = argv[2];
    int do_prune;
    csv_file *train_csv = csv_new(argv[3]);
    csv_file *validate_csv = csv_new(argv[4]);
    csv_file *test_csv = csv_new(argv[5]);
    FILE *prediction_file = fopen(argv[6], "w");

    if(strcmp(split_metric, "entropy") == 0) {
        printf("Using entropy metric for splits\n");
        criterion = CR_ENTROPY;
    }
    else if(strcmp(split_metric, "gini") == 0) {
        printf("Using Gini (population diversity) metric for splits\n");
        criterion = CR_GINI;
    }
    else {
        fprintf(stderr, "Unknown split metric: %s\n", split_metric);
        fprintf(stderr, "Use either 'entropy' or 'gini'\n");
        return 1;
    }

    if(strcmp(prune_str, "prune") == 0) {
        do_prune = 1;
    }
    else if(strcmp(prune_str, "noprune") == 0) {
        do_prune = 0;
    }
    else {
        fprintf(stderr, "Unknown prune directive: %s\n", prune_str);
        fprintf(stderr, "Use either 'prune' or 'noprune'\n");
        return 1;
    }

    if(train_csv == NULL) {
        fprintf(stderr, "Failed to open training CSV file\n");
        return 1;
    }

    if(validate_csv == NULL) {
        fprintf(stderr, "Failed to open validation CSV file\n");
        return 1;
    }

    if(test_csv == NULL) {
        fprintf(stderr, "Failed to open test CSV file\n");
        return 1;
    }

    if(prediction_file == NULL) {
        fprintf(stderr, "Failed to open prediction output file!\n");
        return 1;
    }

    data_set *train_ds = ds_create_from_csv(train_csv, 1);
    csv_free(train_csv);

    data_set *validate_ds = ds_create_from_csv(validate_csv, 1);
    csv_free(validate_csv);

    // 0 in the last arg means no y data
    data_set *test_ds = ds_create_from_csv(test_csv, 0);
    csv_free(test_csv);

    if(train_ds->has_ydata) {
        printf("Training data set has %d rows, %d columns, HAS y data\n",
            train_ds->rowcount, train_ds->colcount);
    }
    else {
        printf("Training data set has %d rows, %d columns, DOES NOT HAVE y data\n", train_ds->rowcount, train_ds->colcount);
    }

    if(validate_ds->has_ydata) {
        printf("Validation data set has %d rows, %d columns, HAS y data\n", validate_ds->rowcount, validate_ds->colcount);
    }
    else {
        printf("Validation data set has %d rows, %d columns, DOES NOT HAVE y data\n", validate_ds->rowcount, validate_ds->colcount);
    }

    if(test_ds->has_ydata) {
        printf("Test data set has %d rows, %d columns, HAS y data\n", test_ds->rowcount, test_ds->colcount);
    }
    else {
        printf("Test data set has %d rows, %d columns, DOES NOT HAVE y data\n", test_ds->rowcount, test_ds->colcount);
    }

    decision_tree *dt = dt_new(0, criterion);

    printf("Training decision tree on training data set...\n");
    if(dt_train(dt, train_ds) == 0) {
        printf("Training successful\n");
    }
    else {
        printf("Training failed?\n");
    }

    printf("Scoring validation data set\n");
    float score = dt_score(dt, validate_ds);
    printf("Score: %.4f\n", score);

    if(do_prune) {
        int precount = dt_node_count(dt);

        printf("Attempting to prune the tree. This may take a while...\n");
        int pruned = dt_prune(dt, validate_ds);
        printf("Pruned %d nodes\n", pruned);

        if(pruned > 0) {
            printf("Calculating score for pruned tree\n");
            float prune_score = dt_score(dt, validate_ds);
            printf("New score: %.4f\n", prune_score);

            printf("Improvement of %.3f\n", prune_score - score);
            printf("Removed %.3f%% of the tree\n", (((float)pruned)/precount)*100);
        }
        else {
            printf("Pruning didn't improve the score...\n");
        }
    }

    printf("Running predictions for test data\n");
    float *preds = dt_predict(dt, test_ds);

    printf("Saving predictions to %s\n", argv[5]);
    fprintf(prediction_file, "Id,Prediction\n");
    for(int i = 0; i < test_ds->rowcount; i++) {
        fprintf(prediction_file, "%d,%d\n", i+1, (int)(preds[i]));
    }

    free(preds);

    printf("Free data sets\n");
    ds_free(train_ds);
    ds_free(validate_ds);
    printf("Free decision tree\n");
    dt_free(dt);

    return 0;
}
