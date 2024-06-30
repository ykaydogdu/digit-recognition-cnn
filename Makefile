CC = g++
CFLAGS = -Wall -g
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# all the object files in OBJDIR are needed to create the executable
digit-recognition: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# compile the source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -I $(INCDIR) -c $< -o $@

run: recognition
	./recognition