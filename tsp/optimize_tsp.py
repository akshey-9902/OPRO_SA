import datetime
import json
import numpy as np
import os
import sys
from absl import app
import google.generativeai as genai
import re
import time

def main(_):
    _GEMINI_API_KEY = '' 
    genai.configure(api_key=_GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Optimization parameters
    num_points = 10          
    num_steps = 20           
    num_decimals = 0         
    num_starting_points = 10 
    num_decode_per_step = 1  

    # Rate limiting parameters
    max_requests_per_minute = 3
    request_interval = 60 / max_requests_per_minute  # Time in seconds between requests

    # Result directory
    datetime_str = datetime.datetime.now().isoformat(timespec='minutes').replace(':', '-')
    save_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # Assuming this is the script's directory
        "outputs",
        "optimization-results",
        f"tsp-gemini-{datetime_str}/",
    )
    os.makedirs(save_folder, exist_ok=True)
    print(f"Result directory: {save_folder}")

    # Instance for TSP
    # # For Random Instances
    # x = np.random.uniform(low=-100, high=100, size=num_points)
    # y = np.random.uniform(low=-100, high=100, size=num_points)
    x = [60, 180, 100, 140, 20, 100, 200, 140, 40, 100,
             180, 60, 120, 180, 20, 100, 200, 20, 60, 160]
    y= [200, 200, 180, 180, 160, 160, 160, 140, 120, 120,
             100, 80, 80, 60, 40, 40, 40, 20, 20, 20]
    x = np.round(x, num_decimals)
    y = np.round(y, num_decimals)

    # Initialize solutions
    initial_solutions = []
    while len(initial_solutions) < num_starting_points:
        sol = np.random.permutation(num_points)
        if sol[0] != 0:  # Ensure all traces start from point 0
            continue
        initial_solutions.append(sol.tolist())

    # Evaluate initial solutions
    previous_traces = []
    for sol in initial_solutions:
        distance = evaluate_distance(x, y, sol)
        trace_str = ','.join(map(str, sol))
        previous_traces.append((trace_str, distance))

    # Start the optimization process
    for step in range(num_steps):
        print(f"\nOptimization Step {step + 1}")

        # Pprompt for the Gemini model
        prompt = create_prompt(x, y, previous_traces, num_points)
        print(f"\nPrompt:\n{prompt}\n")

        # Generate new solutions with rate limiting
        new_traces = []
        for i in range(num_decode_per_step):
            # Enforce rate limiting
            start_time = time.time()

            try:
                response = model.generate_content(prompt)
                response_text = response.text
                print(f"Model Response:\n{response_text}\n")

                new_trace = parse_response(response_text, num_points)
                distance = evaluate_distance(x, y, new_trace)
                print(f'Distance:{distance}')
                trace_str = ','.join(map(str, new_trace))

                if (trace_str, distance) not in previous_traces:
                    previous_traces.append((trace_str, distance))
                    new_traces.append((new_trace, distance))
                    print(f"New trace accepted with distance: {distance}")
                else:
                    print("Duplicate trace or not improved.")

            except ValueError as e:
                print(f"Error parsing response: {e}")
                continue
            except genai.exceptions.APIError as api_err:
                print(f"API error occurred: {api_err}")
                time.sleep(5)  # Wait before retrying
                continue

            # Calculate elapsed time and sleep if necessary
            elapsed_time = time.time() - start_time
            sleep_time = request_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Sort the previous traces to keep only the best ones
        previous_traces = sorted(previous_traces, key=lambda x: x[1])[:50]  # Keep top 50 traces

        # Save the results of this step
        step_results = {
            'step': step + 1,
            'new_traces': [{'trace': trace, 'distance': dist} for trace, dist in new_traces]
        }
        result_path = os.path.join(save_folder, f"results_step_{step + 1}.json")
        with open(result_path, 'w') as file:
            json.dump(step_results, file, indent=4)

def create_prompt(x, y, previous_traces, num_points):
    points_description = "We have the following points with their coordinates:\n"
    for idx, (xi, yi) in enumerate(zip(x, y)):
        points_description += f"{idx}: ({xi}, {yi})\n"

    previous_traces_section = ""
    if previous_traces:
        sorted_traces = sorted(previous_traces, key=lambda x: x[1])
        previous_traces_section = "Here are some previous traces and their total distances (shorter distances are better):\n"
        for idx, (trace_str, distance) in enumerate(sorted_traces, 1):
            previous_traces_section += f"{idx}. Trace: [{trace_str}], Distance: {distance:.2f}\n"

    instructions = f"""
Please provide a new trace different from previous traces that visits all {num_points} points exactly once, starting from point 0, and results in a shorter total distance than the previous traces. The trace should be a list of integers from 0 to {num_points - 1} in the format:

[0, 2, 1, 3, 4, ..., N]

Make sure to include each point exactly once and do not repeat any points. Do not include any explanations or additional text.
""".strip()

    prompt = "\n".join([
        points_description,
        previous_traces_section,
        instructions
    ])

    return prompt

def parse_response(response_text, num_points):
    # Use regular expression to find numbers within square brackets
    match = re.search(r"\[([^\]]+)\]", response_text)
    if match:
        numbers = match.group(1)
        # Split the numbers by comma or whitespace
        parts = re.split(r"[,\s]+", numbers.strip())
        trace = []
        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num < num_points:
                    trace.append(num)
                else:
                    raise ValueError(f"Number out of range: {num}")
            else:
                raise ValueError(f"Invalid number format: {part}")

        if len(trace) != num_points:
            raise ValueError(f"Trace does not contain {num_points} points.")
        if len(set(trace)) != num_points:
            raise ValueError("Trace contains duplicate points.")
        if trace[0] != 0:
            raise ValueError("Trace does not start with point 0.")
        return trace


    raise ValueError("Failed to find a trace in the model's response.")

def evaluate_distance(x, y, trace):
    distance = 0.0
    for i in range(len(trace) - 1):
        idx_current = trace[i]
        idx_next = trace[i + 1]
        dx = x[idx_next] - x[idx_current]
        dy = y[idx_next] - y[idx_current]
        distance += np.hypot(dx, dy)
    # Add distance from the last point back to the starting point
    idx_last = trace[-1]
    idx_start = trace[0]
    dx = x[idx_start] - x[idx_last]
    dy = y[idx_start] - y[idx_last]
    distance += np.hypot(dx, dy)
    return round(distance, 2)

if __name__ == "__main__":
    app.run(main)
