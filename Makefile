pi:
	make -f pierre.mak
ma:
	make -f malik.mak
grid:
	make -f grid.mak

clean:
	rm -f *.o
	rm -f lanczos_modp checker_modp

cleanRec:
	rm -f recordings/*