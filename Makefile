# MOSPCUDA Makefile
#
#   make                      build bin/main (sm_86, host -O3, -lineinfo)
#   make CUDA_ARCH=sm_80      build for another GPU architecture
#   make OPT=                 reproduce the original flags (host code at -O0)
#   make NVCC=/path/to/nvcc   use a specific CUDA toolkit
NVCC      ?= nvcc
CUDA_ARCH ?= sm_86
OPT       ?= -O3
CXXFLAGS  := -std=c++17 -Iheaders
NVFLAGS   := --extended-lambda -arch=$(CUDA_ARCH) $(OPT) -lineinfo
DEPFLAGS  := -MMD -MP
SRCDIR    := src
BINDIR    := bin
BUILDDIR  := build

APP      := $(BINDIR)/main

# Base sources (shared by all targets)
BASE_SRCS := $(SRCDIR)/generateGraph.cu $(SRCDIR)/generateGraphCSR.cu $(SRCDIR)/generateChangedEdges.cu $(SRCDIR)/updateGraphCSR.cu $(SRCDIR)/generateTestCases.cu $(SRCDIR)/Dijkstra.cu $(SRCDIR)/read.cu \
             $(SRCDIR)/csrGraph.cu $(SRCDIR)/stageTimer.cu $(SRCDIR)/validation.cu \
             $(SRCDIR)/changeGenerator.cu $(SRCDIR)/sospUpdateGpu.cu $(SRCDIR)/combinedGraphGpu.cu \
             $(SRCDIR)/deviceGraph.cu $(SRCDIR)/mospUpdate.cu

# Main application (includes sequential SOSP update)
MAIN_SRCS := $(SRCDIR)/main.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/parallelCombinedGraph.cu
MAIN_OBJS := $(MAIN_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Sequential stress test
STRESS_SRCS := $(SRCDIR)/stressTest.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu
STRESS_OBJS := $(STRESS_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Parallel stress test (CUDA)
PARALLEL_STRESS_SRCS := $(SRCDIR)/parallelStressTest.cu $(BASE_SRCS) $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/sequentialSOSPUpdate.cu
PARALLEL_STRESS_OBJS := $(PARALLEL_STRESS_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Oracle tests (new change sets, combined graph, generator/apply checks)
TEST_SRCS := $(SRCDIR)/mospTest.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/parallelCombinedGraph.cu
TEST_OBJS := $(TEST_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Input preparation tool
PREP_SRCS := $(SRCDIR)/mospPrep.cu $(SRCDIR)/changeGenerator.cu $(SRCDIR)/csrGraph.cu $(SRCDIR)/Dijkstra.cu $(SRCDIR)/read.cu
PREP_OBJS := $(PREP_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

# Driver for prepared inputs (benchmarks, validation)
MOSP_SRCS := $(SRCDIR)/mosp.cu $(BASE_SRCS) $(SRCDIR)/sequentialSOSPUpdate.cu $(SRCDIR)/parallelSOSPUpdate.cu $(SRCDIR)/parallelCombinedGraph.cu
MOSP_OBJS := $(MOSP_SRCS:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

.PHONY: all clean run stressTest parallelStressTest test

# Recipes use bash with pipefail so piped test output keeps the exit status.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

all: $(APP) $(BINDIR)/mosp $(BINDIR)/mospPrep $(BINDIR)/mospTest

$(BINDIR) $(BUILDDIR):
	@mkdir -p $@

# --- Main application ---
$(APP): $(MAIN_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) $(DEPFLAGS) -c -o $@ $<

# --- Driver for prepared inputs ---
$(BINDIR)/mosp: $(MOSP_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

$(BINDIR)/mospPrep: $(PREP_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

$(BINDIR)/mospTest: $(TEST_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

# --- Sequential stress test ---
stressTest: $(BINDIR)/stressTest

$(BINDIR)/stressTest: $(STRESS_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

# --- Parallel stress test (CUDA) ---
parallelStressTest: $(BINDIR)/parallelStressTest

$(BINDIR)/parallelStressTest: $(PARALLEL_STRESS_OBJS) | $(BINDIR)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o $@ $^

# --- Tests -------------------------------------------------------------------
# Everything runs inside $(TESTDIR) so the repository stays clean.
#   make test                 stock pipeline + 10 test cases + both stress
#                             tests + the oracle suite (bin/mospTest)
#   make test TEST_SEED=0     stress tests with a random seed (printed)
TESTDIR   := test-output
TEST_SEED ?= 1

test: $(APP) stressTest parallelStressTest $(BINDIR)/mospTest
	@rm -rf $(TESTDIR) && mkdir -p $(TESTDIR)
	@echo "== bin/main (pipeline + 10 generated test cases)"
	@cd $(TESTDIR) && ../$(APP) > main.log 2>&1 || { tail -n 30 main.log; exit 1; }
	@grep "Test Summary" $(TESTDIR)/main.log
	@echo "== bin/stressTest $(TEST_SEED) (sequential SOSP update, 100 random cases)"
	@cd $(TESTDIR) && ../$(BINDIR)/stressTest $(TEST_SEED) > stressTest.log 2>&1 || { grep -E "FAIL|ERROR|Seed" stressTest.log; tail -n 2 stressTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/stressTest.log
	@echo "== bin/parallelStressTest $(TEST_SEED) (parallel SOSP update, 100 random cases)"
	@cd $(TESTDIR) && ../$(BINDIR)/parallelStressTest $(TEST_SEED) > parallelStressTest.log 2>&1 || { grep -E "FAIL|ERROR|Seed" parallelStressTest.log; tail -n 2 parallelStressTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/parallelStressTest.log
	@echo "== bin/mospTest --seed $(TEST_SEED) (oracle suite: change sets, combined graph, thesis example)"
	@$(BINDIR)/mospTest --seed $(TEST_SEED) --work $(TESTDIR)/mospTest > $(TESTDIR)/mospTest.log 2>&1 || { cat $(TESTDIR)/mospTest.log; exit 1; }
	@tail -n 1 $(TESTDIR)/mospTest.log
	@echo "== all tests passed"

clean:
	rm -rf $(BINDIR) $(BUILDDIR) $(TESTDIR)

run: $(APP)
	./$(APP)

# Header dependencies generated by -MMD (rebuild objects when a .cuh changes)
-include $(wildcard $(BUILDDIR)/*.d)
