CC = gcc-10
CFLAGS = -std=c99 -g -Wall -Wextra -Werror  -I. -I/usr/local/Cellar/open-mpi/4.1.0/include 
CFLAGS += -O3 -march=native
LDFLAGS = -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/4.1.0/lib -lmpi
LIBS = -lm
# Uncomment these for OpenMP
#CFLAGS += -fopenmp
#LDFLAGS += -fopenmp

objetsl = mmio.o lanczos_modp.o
objetsc = mmio.o checker_modp.o
all:
checker_modp :	$(objetsc)
				$(CC) $(CFLAGS) $(LDFLAGS) -o checker_modp $(objetsc) -lm

lanczos_modp : 	$(objetsl)
				$(CC) $(CFLAGS) $(LDFLAGS) -o lanczos_modp $(objetsl) -lm

all: lanczos_modp checker_modp
lanczos_modp:  
lanczos_modp.o: lanczos_modp.c mmio.h
checker_modp:    
checker_modp.o: checker_modp.c mmio.h

