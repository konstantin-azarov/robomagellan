SRCS := $(wildcard *.cc)

HDRS := $(wildcard *.hpp)

PREVIEW ?= y

LIBS = -luvc -lopencv_core -lopencv_imgcodecs -lboost_program_options

capture: $(SRCS) $(HDRS)
	g++ $(LIBS) -lopencv_highgui -o bin/capture $(SRCS)

capture_no_preview: $(SRCS) $(HDRS)
	g++ $(LIBS) -DNO_PREVIEW -o bin/capture_no_preview $(SRCS)

all: capture capture_no_preview
