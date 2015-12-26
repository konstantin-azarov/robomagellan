SRCS := $(wildcard *.cc)

HDRS := $(wildcard *.hpp)

bin/capture: $(SRCS) $(HDRS)
	g++ -luvc \
	    -lopencv_core \
			-lopencv_imgcodecs \
			-lopencv_highgui \
			-lboost_program_options \
			-o bin/capture $(SRCS)
