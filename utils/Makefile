SRCS := $(wildcard *.cc)

HDRS := $(wildcard *.hpp)

PREVIEW ?= y

LIBS = -luvc -lopencv_core -lboost_program_options

bin/capture: $(SRCS) $(HDRS)
	g++ -o bin/capture $(SRCS) $(LIBS) -lopencv_highgui

bin/capture_no_preview: $(SRCS) $(HDRS)
	g++ -O2 -DNO_PREVIEW -o bin/capture_no_preview $(SRCS) $(LIBS)

all: bin/capture bin/capture_no_preview
