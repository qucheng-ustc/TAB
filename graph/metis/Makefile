.PHONY: all debug

all: mymetis.out mymetis.so

debug:
	g++ mymetis.cpp $(FLAGS) -g -o mymetis.out

FLAGS=-I/mnt/ssd/qucheng/anaconda3/envs/arrl/include/python3.10\
	  -L/usr/local/lib -L/mnt/ssd/qucheng/anaconda3/envs/arrl/lib\
	  -lmetis -lpython3.10

mymetis.out: mymetis.cpp
	g++ mymetis.cpp $(FLAGS) -o mymetis.out

mymetis.so: mymetis.cpp
	g++ mymetis.cpp $(FLAGS) -fPIC -shared -o mymetis.so 
