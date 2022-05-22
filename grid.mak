CC=mpicc
CFLAGS = -std=c99 -g -Wall -Wextra -Werror  -O3 -march=native -I. #
LDFLAGS =
LIBS = -lm

objetsl = mmio.o lanczos_modp.o
objetsc = mmio.o lanczos_modp.o
all:
checker_modp :	$(objetsc)
				$(CC) $(CFLAGS) $(LDFLAGS) -o checker_modp $(objetsc) -lm

lanczos_modp :	$(objetsl)
				$(CC) $(CFLAGS) $(LDFLAGS) -o lanczos_modp $(objetsl) -lm
