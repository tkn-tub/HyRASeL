# Reproduce the results of the paper in a Docker container (see README.md).
# The Docker image is built automatically when needed.
#
#   make paper                        Simulate and evaluate all configs of the paper, figures -> figures/
#   make simulation  [CONFIG=<name>]  OMNeT++ simulation                  -> simulation/simulations/results/traces/
#   make postprocess [CONFIG=<name>]  Post-processing (localization)      -> evaluation/results/<Config>/postprocessed/
#   make plots       [CONFIG=<name>]  Filtering, statistics and figures   -> evaluation/results/<Config>/<estimator>/<filter>/
#   make evaluation  [CONFIG=<name>]  postprocess + plots
# Without CONFIG, all configs of the paper are used. The scenario is configured in
# simulation/simulations/omnetpp.ini, the evaluation in evaluation/configs/<Config>.yml.
#
# These targets start a container in the background and return. Then:
#   make attach   Watch the progress (tmux; detach with Ctrl-b d, the run continues)
#   make logs     Follow the log (logs/<NAME>.log)
#   make stop     Abort the run
#   make shell    Interactive shell in a new container
#
# Options:
#   CONFIG=<name>  Config of simulation/simulations/omnetpp.ini and evaluation/configs/
#   JOBS=<n>       Parallel simulation runs / post-processing jobs (default: 8 / 4)
#   NAME=<name>    Container name (default: hyrasel), e.g. to run several at the same time

# Configs evaluated in the paper
PAPER_CONFIGS := Baseline SlotNumber AnchorDensity NodeNumber Speed PacketLoss-AnchorNode PacketLoss-NodeNode

CONFIG ?= $(PAPER_CONFIGS)
NAME   ?= hyrasel
IMAGE  := hyrasel-env

DOCKER_ARGS := --user "$$(id -u):$$(id -g)" -e HOME=/tmp -v "$(CURDIR)":/work -w /work

.PHONY: help paper simulation postprocess plots evaluation attach logs stop shell image _start _simulation _postprocess _plots _figures

help:
	@sed -n '2,22s/^# \{0,1\}//p' Makefile

paper:       ; @$(MAKE) --no-print-directory _start TARGET="_simulation _postprocess _plots _figures CONFIG='$(PAPER_CONFIGS)'"
simulation:  ; @$(MAKE) --no-print-directory _start TARGET="_simulation CONFIG='$(CONFIG)'"
postprocess: ; @$(MAKE) --no-print-directory _start TARGET="_postprocess CONFIG='$(CONFIG)'"
plots:       ; @$(MAKE) --no-print-directory _start TARGET="_plots CONFIG='$(CONFIG)'"
evaluation:  ; @$(MAKE) --no-print-directory _start TARGET="_postprocess _plots CONFIG='$(CONFIG)'"

# Start TARGET of this Makefile in a container in the background
_start: image
	@if docker ps -q -f name='^$(NAME)$$' | grep -q .; then \
		echo "Container '$(NAME)' is already running: make attach / make stop, or choose another NAME=<name>" >&2; exit 1; fi
	@mkdir -p logs
	@docker run -d --rm --name $(NAME) $(DOCKER_ARGS) $(IMAGE) ./container.sh logs/$(NAME).log $(TARGET) $(if $(JOBS),JOBS=$(JOBS)) > /dev/null
	@echo "Started in container '$(NAME)': make attach (watch), make logs (log), make stop (abort)"

attach:
	docker exec -it $(NAME) tmux attach -t main

logs:
	tail -f logs/$(NAME).log

stop:
	docker stop $(NAME)

# The container does not know your user name, so the shell is started without
# the Ubuntu startup files and with its own prompt
shell: image
	@docker run --rm -it $(DOCKER_ARGS) -e PS1='[container] \w$$ ' $(IMAGE) bash --norc

# Build the image if it does not exist or the Dockerfile / requirements changed (cached otherwise)
image:
	@docker image inspect $(IMAGE) > /dev/null 2>&1 || echo "Building the Docker image (first time ~10 min) ..."
	@docker build -q -t $(IMAGE) . > /dev/null

# --- Run inside the container -----------------------------------------------------

_simulation:
	$(MAKE) -C simulation makefiles all
	@for config in $(CONFIG); do $(MAKE) -C simulation run CONFIG=$$config || exit 1; done

_postprocess:
	$(MAKE) -C evaluation postprocess CONFIG="$(patsubst %,configs/%.yml,$(CONFIG))"

_plots:
	$(MAKE) -C evaluation analyze CONFIG="$(patsubst %,configs/%.yml,$(CONFIG))"

# Collect the figures of all configs
_figures:
	mkdir -p figures
	cp evaluation/results/*/*/*/figures/*.pdf figures/
