#!/bin/bash

echo "Fixed installation errors"
pip freeze | xargs pip uninstall -y
