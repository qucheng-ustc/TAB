.PHONY: all debug clean

all: mymetis.out mymetis.so

FLAGS=-I${HOME}/anaconda3/envs/arrl/include/python3.10\
	  -L/usr/local/lib -L${HOME}/anaconda3/envs/arrl/lib\
	  -lmetis -lpython3.10

debug:
	g++ mymetis.cpp $(FLAGS) -g -o mymetis.out

mymetis.out: mymetis.cpp
	g++ mymetis.cpp $(FLAGS) -o mymetis.out

mymetis.so: mymetis.cpp
	g++ mymetis.cpp $(FLAGS) -fPIC -shared -o mymetis.so 

clean:
	- rm *.so
	- rm -r build
	- rm -r MyMetis.egg-info

