CC=c99
CFLAGS=-g -Wall
LDFLAGS=-lm

all: decisiontree

decisiontree: main.o csv.o decision_tree.o data_set.o
	$(CC) main.o csv.o decision_tree.o data_set.o -o dt_main $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

csv.o: csv.h csv.c
	$(CC) $(CFLAGS) -c csv.c

data_set.o: data_set.h data_set.c
	$(CC) $(CFLAGS) -c data_set.c

decision_tree.o: decision_tree.h decision_tree.c
	$(CC) $(CFLAGS) -c decision_tree.c

clean:
	rm -f *.o dt_main
