# MOSPCUDA Makefile
NVCC     := nvcc
CXXFLAGS := -std=c++17 -Iheaders
CUDA_ARCH ?= sm_70
NVFLAGS  := --extended-lambda -arch=$(CUDA_ARCH)
SRCDIR   := src
BINDIR   := bin
BUILDDIR := build

APP      := $(BINDIR)/main

# Base sources (shared by all targets)
BASE_SRCS := $(SRCDIR)/generateGraph.cu $(SRCDIR)/generateGraphCSR.cu $(SRCDIR)/generateChangedEdges.cu $(SRCDIR)/updateGraphCSR.cu $(SRCDIR)/generateTestCases.cu $(SRCDIR)/Dijkstra.cu $(SRCDIR)/read.cu

# Main application (includes sequential SOSP update)
MAIN_SRCS := $(SRCDIR)/main.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/parallelCombinedGraph.cu
MAIN_OBJS := $(MAIN_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Sequential stress test
STRESS_SRCS := $(SRCDIR)/stressTest.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu
STRESS_OBJS := $(STRESS_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Parallel stress test (CUDA)
PARALLEL_STRESS_SRCS := $(SRCDIR)/parallelStressTest.cu $(BASE_SRCS) $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/sequentialSOSPUpdate.cu
PARALLEL_STRESS_OBJS := $(PARALLEL_STRESS_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/parallel_%.o)

.PHONY: all clean run stressTest parallelStressTest

all: $(APP)

$(BINDIR) $(BUILDDIR):
	@mkdir -p $@

# --- Main application ---
$(APP): $(MAIN_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c -o $@ $<

# --- Sequential stress test ---
stressTest: $(BINDIR)/stressTest

$(BINDIR)/stressTest: $(STRESS_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

# --- Parallel stress test (CUDA) ---
parallelStressTest: $(BINDIR)/parallelStressTest

$(BINDIR)/parallelStressTest: $(PARALLEL_STRESS_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

$(BUILDDIR)/parallel_%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf $(BINDIR) $(BUILDDIR)

run: $(APP)
	./$(APP)
