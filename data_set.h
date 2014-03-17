#pragma once

#include "csv.h"

typedef struct data_set {
    unsigned int colcount;
    unsigned int rowcount;
    // not for external use
    unsigned int rowcapacity;
    int has_ydata;
    float **x_data;
    float *y_data;
} data_set;

// colcount must be the same for all rows
// has_ydata determines if the dataset has y values
data_set* ds_new(unsigned int colcount, int has_ydata);

// creates a new data set from the specified csv
data_set* ds_create_from_csv(csv_file *csv, int last_row_is_y);

// free the dataset
void ds_free(data_set *ds);

// x should be an array of floats with length `colcount`
// y will be ignored if has_ydata is false
void ds_add_item(data_set *ds, float *x, float y);

// compute the min/max/mean/variance of the specified column
float ds_col_mean(data_set *ds, unsigned int col);
float ds_col_variance(data_set *ds, unsigned int col);
float ds_col_min(data_set *ds, unsigned int col);
float ds_col_max(data_set *ds, unsigned int col);

// compute the entropy in the data set
float ds_entropy(data_set *ds);

// population diversity (Gini Index)
// sum of squares of proportions of classes
float ds_gini(data_set *ds);

// return an array of all of the classes in y_data, and sets count to the
// number of classes
// WARNING: this breaks if there are more than 1024 classes, but I'm too lazy
// to do it right
float* ds_classes(data_set *ds, int *count);
