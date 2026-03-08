#!/bin/bash
brainsurgery 'inputs[0]=models/gpt2' output.path=models/teacst 'transforms[0].dump.target=.*' $@
