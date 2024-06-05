# Define the number of test cases.
NUM_TEST_CASES=$1

# Define the Python script filename.
PYTHON_SCRIPT="../ip.py"
OUTPUT_DIR="../results-ip"

# Loop over all test cases.
for TEST_CASE in "$@"; do
    # Define the output directory and files for the current test case.
    OUTPUT_FILE="${OUTPUT_DIR}/test_case_${TEST_CASE}.txt"
    PLOT_INPUT="${OUTPUT_DIR}/test_input_${TEST_CASE}.png"
    PLOT_OUTPUT="${OUTPUT_DIR}/test_output_${TEST_CASE}.png"
    # Run the Python script with the test case number and save its output to a file.
    python "$PYTHON_SCRIPT" "Example$TEST_CASE.xlsx" 10000 -v > "$OUTPUT_FILE"

    # Move the generated plot images to the desired locations
    if [ -f "input.png" ]; then
        mv "input.png" "$PLOT_INPUT"
    else
        echo "Plot Input not found for test case $TEST_CASE!"
    fi

    if [ -f "output.png" ]; then
        mv "output.png" "$PLOT_OUTPUT"
    else
        echo "Plot Output not found for test case $TEST_CASE!"
    fi

    echo "Test case $TEST_CASE execution completed."
done

echo "All test cases execution completed."

# Pause to keep the PowerShell window open
read -p "Press any key to continue..."