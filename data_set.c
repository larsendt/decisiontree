#include "data_set.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void ds_resize(data_set *ds);

data_set* ds_new(unsigned int colcount, int has_ydata) {
    data_set *ds = malloc(sizeof(data_set));
    ds->colcount = colcount;
    ds->has_ydata = has_ydata;
    ds->rowcount = 0;
    ds->rowcapacity = 0;
    ds->x_data = NULL;
    ds->y_data = NULL;
    return ds;
}

data_set* ds_create_from_csv(csv_file *csv, int last_row_is_y) {
    data_set *ds;
    if(last_row_is_y) {
        ds = ds_new(csv->colcount-1, 1);
    }
    else {
        ds = ds_new(csv->colcount, 0);
    }

    ds->rowcount = csv->rowcount;
    ds->rowcapacity = csv->rowcount;

    ds->x_data = malloc(ds->rowcount * sizeof(float*));
    if(last_row_is_y) {
        ds->y_data = malloc(ds->rowcount * sizeof(float));
    }

    for(int i = 0; i < ds->rowcount; i++) {
        ds->x_data[i] = malloc(ds->colcount * sizeof(float));
        memcpy(ds->x_data[i], csv->data[i], ds->colcount * sizeof(float));
        if(last_row_is_y) {
            ds->y_data[i] = csv->data[i][csv->colcount-1];
        }
    }

    return ds;
}

void ds_free(data_set *ds) {
    for(int i = 0; i < ds->rowcapacity; i++) {
        free(ds->x_data[i]);
    }

    free(ds->x_data);
    free(ds->y_data);
}

void ds_resize(data_set *ds) {
    if(ds->rowcount < ds->rowcapacity) {
        return;
    }

    ds->rowcapacity += 10;

    ds->x_data = realloc(ds->x_data, ds->rowcapacity * sizeof(float*));
    ds->y_data = realloc(ds->y_data, ds->rowcapacity * sizeof(float));

    for(int i = ds->rowcount; i < ds->rowcapacity; i++) {
        ds->x_data[i] = malloc(ds->colcount * sizeof(float));
    }
}

void ds_add_item(data_set *ds, float *x, float y) {
    ds_resize(ds);

    memcpy(ds->x_data[ds->rowcount], x, ds->colcount * sizeof(float));
    ds->y_data[ds->rowcount] = y;

    ds->rowcount += 1;
}


float ds_col_mean(data_set *ds, unsigned int col) {
    double mean = 0;
    for(int i = 0; i < ds->rowcount; i++) {
        mean += ds->x_data[i][col];
    }

    return (float)(mean / ds->rowcount);
}

float ds_col_variance(data_set *ds, unsigned int col) {
    float mean = ds_col_mean(ds, col);
    double variance = 0;
    for(int i = 0; i < ds->rowcount; i++) {
        double diff = mean - ds->x_data[i][col];
        variance += diff;
    }
    return (float)(variance / ds->rowcount);
}

float ds_col_min(data_set *ds, unsigned int col) {
    float min = ds->x_data[0][col];
    for(int i = 0; i < ds->rowcount; i++) {
        if(ds->x_data[i][col] < min) {
            min = ds->x_data[i][col];
        }
    }
    return min;
}

float ds_col_max(data_set *ds, unsigned int col) {
    float max = ds->x_data[0][col];
    for(int i = 0; i < ds->rowcount; i++) {
        if(ds->x_data[i][col] > max) {
            max = ds->x_data[i][col];
        }
    }
    return max ;
}

float* ds_classes(data_set *ds, int *count) {
    *count = 0;
    float *classes = malloc(1024 * sizeof(float));

    for(int i = 0; i < ds->rowcount; i++) {
        float val = ds->y_data[i];
        int flag = 0;
        for(int j = 0; j < *count; j++) {
            if(val == classes[j]) {
                flag = 1;
            }
        }
        if(!flag) {
            classes[*count] = val;
            *count += 1;
        }
    }
    return classes;
}

float ds_entropy(data_set *ds) {
    if(!ds->has_ydata) {
        fprintf(stderr, "Entropy calculation requires Y data!\n");
        return 0.0;
    }

    int total = ds->rowcount;
    int classcount = 0;
    float *classes = ds_classes(ds, &classcount);
    int *classcounts = malloc(classcount * sizeof(int));
    memset(classcounts, 0, classcount * sizeof(int));

    for(int i = 0; i < ds->rowcount; i++) {
        float val = ds->y_data[i];
        for(int j = 0; j < classcount; j++) {
            if(val == classes[j]) {
                classcounts[j] += 1;
            }
        }
    }

    float entropy = 0.0;
    for(int i = 0; i < classcount; i++) {
        float frac = ((float)classcounts[i]) / total;
        float logfrac = 0.0;
        if(frac != 0.0) {
            logfrac = log(frac) / log(2);
        }
        entropy += frac * logfrac;
    }
    entropy *= -1;

    free(classes);
    free(classcounts);
    return entropy;
}


float ds_gini(data_set *ds) {
    if(!ds->has_ydata) {
        fprintf(stderr, "Gini Index calculation requires Y data!\n");
        return 0.0;
    }

    int total = ds->rowcount;
    int classcount = 0;
    float *classes = ds_classes(ds, &classcount);
    int *classcounts = malloc(classcount * sizeof(int));
    memset(classcounts, 0, classcount * sizeof(int));

    for(int i = 0; i < ds->rowcount; i++) {
        float val = ds->y_data[i];
        for(int j = 0; j < classcount; j++) {
            if(val == classes[j]) {
                classcounts[j] += 1;
            }
        }
    }

    float gini = 0.0;
    for(int i = 0; i < classcount; i++) {
        float frac = ((float)classcounts[i]) / total;
        gini += frac * frac;
    }

    free(classes);
    free(classcounts);
    return gini;
}


