CC=mpicc
CFLAGS = -std=c99 -g -Wall -Wextra -Werror  -O3 -march=native -I. #
LDFLAGS =
LIBS = -lm

objetsl = mmio.o lanczos_modp.o
objetsc = mmio.o checker_modp.o
all: lanczos_modp checker_modp 
checker_modp :	$(objetsc)
				gcc $(CFLAGS) $(LDFLAGS) -o checker_modp $(objetsc) -lm

lanczos_modp :	$(objetsl)
				$(CC) $(CFLAGS) $(LDFLAGS) -o lanczos_modp $(objetsl) -lm
