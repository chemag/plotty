all: none


plotty_pb2.py: plotty.proto
	protoc -I=. --python_out=./ plotty.proto

clean:
	rm -f *pyc

realclean: clean
	rm -f plotty_pb2.py
