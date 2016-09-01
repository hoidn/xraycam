#libclusters.so : clusters.o
#	gcc  -shared -Wl,-soname,libclusters.so -o libclusters.so clusters.o

CFLAGS=

libclusters.so : clusters.o arrayqueue.o
	gcc $(CFLAGS) -shared -Wl,-soname,libclusters.so -o libclusters.so clusters.o arrayqueue.o

clusters.o : clusters.c
	gcc $(CFLAGS) -c -fPIC clusters.c -o clusters.o

arrayqueue.o : arrayqueue.c
	gcc $(CFLAGS) -c -fPIC arrayqueue.c -o arrayqueue.o

clean :
	-rm -vf libclusters.so clusters.o cluster
