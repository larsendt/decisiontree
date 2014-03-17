#pragma once

typedef struct csv_file {
    char *filename;
    unsigned int rowcount;
    unsigned int colcount;
    float **data;
} csv_file;


// read from the specified file
// requires columns separated by commas, with no whitespace
// assumes all columns are floats
csv_file* csv_new(char *filename);
void csv_free(csv_file *csv);

// calculate the (min|max|mean|var) of the specified column
float csv_col_min(csv_file *csv, int col);
float csv_col_max(csv_file *csv, int col);
float csv_col_mean(csv_file *csv, int col);
float csv_col_variance(csv_file *csv, int col);

