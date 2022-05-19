CC=mpicc
CFLAGS = -std=c99 -g -Wall -Wextra -Werror  -O3 -march=native -I. #
LDFLAGS =
LIBS = -lm

objets = mmio.o lanczos_modp.o mmio.o checker_modp.o
lanczos_modp : $(objets)
	$(CC) $(CFLAGS) -o lanczos_modp $(objets) -lm

# Uncomment these for OpenMP
#CFLAGS += -fopenmp
#LDFLAGS += -fopenmp

all: lanczos_modp checker_modp
lanczos_modp: mmio.o lanczos_modp.o
lanczos_modp.o: lanczos_modp.c mmio.h
checker_modp:   mmio.o checker_modp.o
checker_modp.o: checker_modp.c mmio.h

