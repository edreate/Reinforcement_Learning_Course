format:
	clear
	@echo "Formatting python code"
	ruff format . --line-length 120

format-cpp:
	clear
	@echo "Formatting c++ code"
	find src/ \( -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} +