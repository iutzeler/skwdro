#!/bin/bash


for filename in tests/misc/test_*.py
do
	echo "Test: $filename"
	python -c "print('#' * $((${#filename}+6)))"
	pytest $filename
done
