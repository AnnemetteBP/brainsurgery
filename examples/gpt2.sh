#!/bin/bash
brainsurgery 'inputs[0]=models/gpt2' 'transforms[0].dump.format=compact' $@
