CFLAGS = -std=c99 -g -Wall -Wextra -Werror  -I. -I/usr/local/Cellar/open-mpi/4.1.0/include 
CFLAGS += -O3 -march=native
LDFLAGS = -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/4.1.0/lib -lmpi
# Uncomment these for OpenMP
#CFLAGS += -fopenmp
#LDFLAGS += -fopenmp
all: lanczos_modp checker_modp
lanczos_modp: mmio.o lanczos_modp.o
lanczos_modp.o: lanczos_modp.c mmio.h
checker_modp:   mmio.o checker_modp.o
checker_modp.o: checker_modp.c mmio.h
