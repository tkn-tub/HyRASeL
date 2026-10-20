# Evaluation pipeline: post-processing -> statistics extraction -> figures.
# Every evaluation is configured by configs/<Config>.yml (defaults: configs/defaults.yml).
#
#   make <step> CONFIG=configs/Speed.yml     any config file
#   make <step>-Speed                        short form for configs/Speed.yml
#   make <step>-all                          all configs
#
# Steps: postprocess, extract, plots, analyze (extract + plots), full (all three).
# All steps except plots start their own tmux session (attach command is printed).
#
# Optional overrides, e.g. make full-Speed JOBS=12 REPS=0-4 PLOT_ARGS="--format png":
#   JOBS, REPS, METHODS, POSTPROCESS_ARGS, EXTRACT_ARGS, PLOT_ARGS,
#   FOREGROUND=1 (no tmux session), DRY_RUN=1 (only print the commands),
#   RESUME=1 (skip post-processing jobs that are already done)

# Python of the virtual environment if it exists (see README), otherwise from PATH
PYTHON   ?= $(if $(wildcard .venv/bin/python3),.venv/bin/python3,python3)
PIPELINE := $(PYTHON) src/pipeline.py
CONFIGS  := $(filter-out configs/defaults.yml,$(wildcard configs/*.yml))
STEPS    := postprocess extract plots analyze full

OPTIONS := $(if $(JOBS),--jobs $(JOBS)) \
           $(if $(REPS),--reps $(REPS)) \
           $(if $(METHODS),--methods $(METHODS)) \
           $(if $(POSTPROCESS_ARGS),--postprocess-args "$(POSTPROCESS_ARGS)") \
           $(if $(EXTRACT_ARGS),--extract-args "$(EXTRACT_ARGS)") \
           $(if $(PLOT_ARGS),--plot-args "$(PLOT_ARGS)") \
           $(if $(filter 1,$(FOREGROUND)),--foreground) \
           $(if $(filter 1,$(DRY_RUN)),--dry-run) \
           $(if $(filter 1,$(RESUME)),--resume)

.PHONY: help $(STEPS) $(addsuffix -all,$(STEPS))

help:
	@echo "Usage: make <step> CONFIG=configs/<name>.yml | make <step>-<name> | make <step>-all"
	@echo "Steps:   $(STEPS)"
	@echo "Configs: $(notdir $(basename $(CONFIGS)))"
	@echo "See the header of this Makefile for optional overrides."

$(STEPS):
	@test -n "$(CONFIG)" || { echo "Missing CONFIG=configs/<name>.yml"; exit 1; }
	$(PIPELINE) $@ $(CONFIG) $(OPTIONS)

$(addsuffix -all,$(STEPS)):
	$(PIPELINE) $(@:-all=) $(CONFIGS) $(OPTIONS)

postprocess-%: ; $(PIPELINE) postprocess configs/$*.yml $(OPTIONS)
extract-%:     ; $(PIPELINE) extract configs/$*.yml $(OPTIONS)
plots-%:       ; $(PIPELINE) plots configs/$*.yml $(OPTIONS)
analyze-%:     ; $(PIPELINE) analyze configs/$*.yml $(OPTIONS)
full-%:        ; $(PIPELINE) full configs/$*.yml $(OPTIONS)
