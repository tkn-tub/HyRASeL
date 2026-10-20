# Build and run the simulations (also used by the OMNeT++ IDE to build the project).
#
#   make                       Build the simulation library
#   make run CONFIG=<Config>   Run all runs of one config of simulations/omnetpp.ini
#   make run-all               Run all configs of simulations/omnetpp.ini
#   make list                  List the configs and their number of runs
#   make clean                 Remove the build output
#
# Traces are written to simulations/results/traces/<Config>/.
#
# Options:
#   INET_DIR=<path>      INET 4.5.4 installation (default: ../inet-4.5.4, i.e. next to this project)
#   OMNETPP_ROOT=<path>  OMNeT++ 6.2.0 installation (not needed if its setenv was sourced)
#   JOBS=<n>             Number of parallel runs (default: 8)
#   RUNS=<filter>        Only some runs: run numbers (RUNS=0..3) or a filter (RUNS='$repetition==0')

INET_DIR ?= ../inet-4.5.4
JOBS     ?= 8
MODE     ?= release

INET    := $(abspath $(INET_DIR))
CONFIGS := $(shell sed -n 's/^\[Config \(.*\)\]/\1/p' simulations/omnetpp.ini)
ifdef OMNETPP_ROOT
export PATH := $(OMNETPP_ROOT)/bin:$(PATH)
endif

OPP_RUN := opp_run -l ../src/localization -l $(INET)/src/INET -n .:../src:$(INET)/src -u Cmdenv

.PHONY: all makefiles clean run run-all list check-env

all: src/Makefile
	$(MAKE) -C src MODE=$(MODE)

# src/Makefile is generated (the OMNeT++ IDE generates it from .oppbuildspec)
src/Makefile:
	$(MAKE) makefiles

makefiles: check-env
	cd src && opp_makemake -f --deep --make-so -o localization -KINET_PROJ=$(INET) -DINET_IMPORT -I'$$(INET_PROJ)/src' -L'$$(INET_PROJ)/src' -l'INET$$(D)'

clean:
	-$(MAKE) -C src clean

check-env:
	@command -v opp_run >/dev/null || { echo "OMNeT++ not found: source <omnetpp>/setenv or set OMNETPP_ROOT=<path>" >&2; exit 1; }
	@test -d $(INET)/src/inet || { echo "INET not found in $(INET): set INET_DIR=<path to INET 4.5.4>" >&2; exit 1; }

run: check-env
	@test -n "$(CONFIG)" || { echo "Usage: make run CONFIG=<Config>, one of: $(CONFIGS)" >&2; exit 1; }
	cd simulations && opp_runall -j$(JOBS) $(OPP_RUN) -c $(CONFIG) $(if $(RUNS),-r '$(value RUNS)')

run-all:
	@for config in $(CONFIGS); do $(MAKE) run CONFIG=$$config || exit 1; done

list: check-env
	@cd simulations && for config in $(CONFIGS); do echo "$$config: $$($(OPP_RUN) -c $$config -q runs | sed -n "s/^Number of runs: //p") runs"; done
