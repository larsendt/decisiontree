#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void split_line(char *line, float *floatbuf, int buflen, int *ncols);
int count_columns(FILE *csvfile);

csv_file* csv_new(char *filename) {
    int len = strlen(filename);
    csv_file *csv = malloc(sizeof(csv_file));
    csv->filename = malloc(len+1);
    memcpy(csv->filename, filename, len+1);

    FILE *f = fopen(csv->filename, "r");

    if(!f) {
        fprintf(stderr, "Unable to open file '%s'\n", csv->filename);
        return NULL;
    }


    csv->colcount = count_columns(f);
    csv->rowcount = 0;

    char buf[4096];
    int ncols;
    int curlines = 1;
    float **data = malloc(curlines * sizeof(float*));
    data[0] = malloc(csv->colcount * sizeof(float));
    while(!feof(f)) {
        if(fscanf(f, "%4096s", buf) != 1) {
            break;
        }

        if(csv->rowcount >= curlines) {
            data = realloc(data, curlines*2*sizeof(float*));
            for(int i = curlines; i < curlines*2; i++) {
                data[i] = malloc(csv->colcount * sizeof(float));
            }
            curlines *= 2;
        }

        split_line(buf, data[csv->rowcount], csv->colcount, &ncols);
        csv->rowcount += 1;

        if(ncols != csv->colcount) {
            fprintf(stderr, "Warning! Expected %d columns, got %d on line %d\n",
                    csv->colcount, ncols, csv->rowcount);
        }
    }

    csv->data = malloc(csv->rowcount * sizeof(float*));
    for(int i = 0; i < csv->rowcount; i++) {
        csv->data[i] = data[i];
    }

    for(int i = csv->rowcount; i < curlines; i++) {
        free(data[i]);
    }
    free(data);
    fclose(f);
    return csv;
}

void csv_free(csv_file *csv) {
    if(csv == NULL ) {
        return;
    }

    for(int i = 0; i < csv->rowcount; i++) {
        free(csv->data[i]);
    }

    free(csv->data);
    free(csv->filename);
    free(csv);
}


float csv_col_min(csv_file *csv, int col) {
    if(col >= csv->colcount) {
        fprintf(stderr, "csv_col_min got an invalid column: %d (range is 0-%d)\n", col, csv->colcount-1);
        return 0;
    }

    float min = csv->data[0][col];
    for(int i = 0; i < csv->rowcount; i++) {
        if(csv->data[i][col] < min) {
            min = csv->data[i][col];
        }
    }
    return min;
}

float csv_col_max(csv_file *csv, int col) {
    if(col >= csv->colcount || col < 0) {
        fprintf(stderr, "csv_col_max got an invalid column: %d (range is 0-%d)\n", col, csv->colcount-1);
        return 0;
    }

    float max = csv->data[0][col];
    for(int i = 0; i < csv->rowcount; i++) {
        if(csv->data[i][col] > max) {
            max = csv->data[i][col];
        }
    }
    return max;
}


float csv_col_mean(csv_file *csv, int col) {
    if(col >= csv->colcount || col < 0) {
        fprintf(stderr, "csv_col_mean got an invalid column: %d (range is 0-%d)\n", col, csv->colcount-1);
        return 0;
    }

    float mean = 0;
    for(int i = 0; i < csv->rowcount; i++) {
        mean += csv->data[i][col];
    }
    return mean / csv->rowcount;
}

float csv_col_variance(csv_file *csv, int col) {
    if(col >= csv->colcount || col < 0) {
        fprintf(stderr, "csv_col_mean got an invalid column: %d (range is 0-%d)\n", col, csv->colcount-1);
        return 0;
    }

    float mean = csv_col_mean(csv, col);
    float variance = 0;
    for(int i = 0; i < csv->rowcount; i++) {
        float diff = mean - csv->data[i][col];
        variance += diff * diff;
    }
    return variance / csv->rowcount;
}


void split_line(char *line, float *buf, int buflen, int *ncols) {
    float val;
    char *pos = line;
    char *tmppos;
    *ncols = 0;
    while(sscanf(pos, "%f,", &val) == 1) {
        if(*ncols == buflen) {
            return;
        }
        buf[*ncols] = val;

        tmppos = strchr(pos, ',');
        if(tmppos == NULL) {
            break;
        }
        else {
            pos = tmppos + 1;
        }

        *ncols += 1;
    }

    if(*ncols <= buflen) {
        if(sscanf(pos, "%f", &val) == 1) {
            buf[*ncols] = val;
            *ncols += 1;
        }
    }
}


int count_columns(FILE *csvfile) {
    char line[4096];

    if(fscanf(csvfile, "%4096s", line) != 1) {
        fseek(csvfile, 0, SEEK_SET);
        return 0;
    }

    int c = 0;
    for(int i = 0; i < strlen(line); i++) {
        if(line[i] == ',') {
            c += 1;
        }
    }
    fseek(csvfile, 0, SEEK_SET);
    // turns out that little +1 is really important...
    return c+1;
}
