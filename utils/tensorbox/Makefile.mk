SHELL := /bin/bash

.PHONY: all
all:
	cd /opt/anaconda3/bin/ #path to anaconda /bin folder, had to be manually changed, or excluded#
	pip install runcython3
	makecython3++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"